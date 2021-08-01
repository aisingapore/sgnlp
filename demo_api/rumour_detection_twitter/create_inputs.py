def generate_structure(thread_len, max_posts):
    time_delay_ids = [0] * thread_len + [1] * (max_posts - thread_len)

    structure_ids = [
        [3] * idx + [4] + [2] * (thread_len - 1 - idx) + [5] * (max_posts - thread_len)
        for idx in range(thread_len)
    ] + [[5] * max_posts] * (max_posts - thread_len)

    post_attention_mask = [1] * thread_len + [0] * (max_posts - thread_len)

    return [time_delay_ids], [structure_ids], [post_attention_mask]
