#include "models.h"

ggml_cgraph * clip_graph_siglip::build() {
    ggml_tensor * inp = build_inp();

    ggml_tensor * learned_pos_embd = model.position_embeddings;
    if (proj_type == PROJECTOR_TYPE_LFM2 || proj_type == PROJECTOR_TYPE_PHI4) {
        learned_pos_embd = resize_position_embeddings();
    }

    ggml_tensor * cur = build_vit(
                            inp, n_patches,
                            NORM_TYPE_NORMAL,
                            hparams.ffn_op,
                            learned_pos_embd,
                            nullptr);

    if (proj_type == PROJECTOR_TYPE_GEMMA3) {
        const int batch_size = 1;
        GGML_ASSERT(n_patches_x == n_patches_y);
        const int patches_per_image = n_patches_x;
        const int kernel_size = hparams.n_merge;

        cur = ggml_transpose(ctx0, cur);
        cur = ggml_cont_4d(ctx0, cur, patches_per_image, patches_per_image, n_embd, batch_size);

        // doing a pool2d to reduce the number of output tokens
        cur = ggml_pool_2d(ctx0, cur, GGML_OP_POOL_AVG, kernel_size, kernel_size, kernel_size, kernel_size, 0, 0);
        cur = ggml_reshape_3d(ctx0, cur, cur->ne[0] * cur->ne[0], n_embd, batch_size);
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

        // apply norm before projection
        cur = ggml_rms_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, model.mm_soft_emb_norm_w);

        // apply projection
        cur = ggml_mul_mat(ctx0,
            ggml_cont(ctx0, ggml_transpose(ctx0, model.mm_input_proj_w)),
            cur);

    } else if (proj_type == PROJECTOR_TYPE_IDEFICS3) {
        // pixel_shuffle
        // https://github.com/huggingface/transformers/blob/0a950e0bbe1ed58d5401a6b547af19f15f0c195e/src/transformers/models/idefics3/modeling_idefics3.py#L578
        const int scale_factor = model.hparams.n_merge;
        cur = build_patch_merge_permute(cur, scale_factor);
        cur = build_mm(model.projection, cur);

    } else if (proj_type == PROJECTOR_TYPE_LFM2) {
        // pixel unshuffle block
        const int scale_factor = model.hparams.n_merge;
        cur = build_patch_merge_permute(cur, scale_factor);

        // projection, in LFM2-VL input norm is optional
        if (model.mm_input_norm_w) {
            cur = ggml_norm(ctx0, cur, 1e-5); // default nn.LayerNorm
            cur = ggml_mul(ctx0, cur, model.mm_input_norm_w);
        }

        if (model.mm_input_norm_b) {
            cur = ggml_add(ctx0, cur, model.mm_input_norm_b);
        }

        cur = build_ffn(cur,
            model.mm_1_w, model.mm_1_b,
            nullptr, nullptr,
            model.mm_2_w, model.mm_2_b,
            FFN_GELU,
            -1);

    } else if (proj_type == PROJECTOR_TYPE_JANUS_PRO) {
        cur = build_ffn(cur,
            model.mm_0_w, model.mm_0_b,
            nullptr, nullptr,
            model.mm_1_w, model.mm_1_b,
            hparams.ffn_op,
            -1);

    } else if (proj_type == PROJECTOR_TYPE_PHI4) {
        cur = build_ffn(cur,
            model.mm_0_w, model.mm_0_b,
            nullptr, nullptr,
            model.mm_2_w, model.mm_2_b,
            FFN_GELU,
            -1);

    } else {
        GGML_ABORT("SigLIP: Unsupported projector type");
    }

    // build the graph
    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * clip_graph_ovis::build() {
    // 2D pos inputs for RoPE
    ggml_tensor * pos_h = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_h, "pos_h");
    ggml_set_input(pos_h);

    ggml_tensor * pos_w = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_w, "pos_w");
    ggml_set_input(pos_w);

    // SigLIP2 NaViT uses emb = cat(h,w,h,w) with rotate_half pairing (i, i+d/2)
    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        return build_rope_2d_ovis(ctx0, cur, pos_h, pos_w, hparams.rope_theta);
    };

    ggml_tensor * inp = build_inp();

    // Ovis SigLIP2 uses BOTH learned position embeddings AND RoPE.
    // Learned PE (with interpolation for dynamic sizes) is added to patch embeddings;
    // RoPE is applied in attention...
    ggml_tensor * learned_pos_embd = model.position_embeddings
        ? resize_position_embeddings()
        : nullptr;

    ggml_tensor * cur = build_vit(
                            inp, n_patches,
                            NORM_TYPE_NORMAL,
                            hparams.ffn_op,
                            learned_pos_embd,
                            add_pos);
    // cur: [n_embd, n_patches]

    // 2. Hidden stride merge (2x2 pixel unshuffle)
    const int scale_factor = hparams.n_merge; // hidden_stride = 2
    cur = build_patch_merge_permute(cur, scale_factor);
    // cur: [n_embd * scale_factor^2, n_merged_patches]

    // 3. Linear projection (visual_tokenizer.head.0)
    cur = ggml_mul_mat(ctx0, model.mm_0_w, cur);
    // cur: [n_visual_logits, n_merged_patches]

    // 4. LayerNorm (visual_tokenizer.head.1)
    cur = ggml_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, model.mm_post_norm_w);
    cur = ggml_add(ctx0, cur, model.mm_post_norm_b);

    // 5. Softmax over visual vocabulary dimension
    cur = ggml_soft_max(ctx0, cur);

    // 6. Soft token embedding lookup via VTE
    //    result = softmax_probs @ VTE_sub (matrix multiply)
    //    softmax_probs: [n_visual_logits, n_merged_patches]
    //    VTE (ggml):    [n_embd_text, visual_vocab_size]
    //    We only use the first n_visual_logits rows of VTE (skip indicator tokens)
    const int n_visual_logits = model.mm_0_w->ne[1];
    ggml_tensor * vte_sub = ggml_view_2d(ctx0, model.mm_vte_w,
        model.mm_vte_w->ne[0], n_visual_logits,
        model.mm_vte_w->nb[1], 0);
    // vte_sub: [n_embd_text, n_visual_logits]

    // Transpose VTE so contraction dimension is ne[0]
    ggml_tensor * vte_t = ggml_cont(ctx0, ggml_transpose(ctx0, vte_sub));
    // vte_t: [n_visual_logits, n_embd_text]

    cur = ggml_mul_mat(ctx0, vte_t, cur);
    // cur: [n_embd_text, n_merged_patches]

    ggml_build_forward_expand(gf, cur);

    return gf;
}
