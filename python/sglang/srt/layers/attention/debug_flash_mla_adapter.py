def flash_mla_with_kvcache_entrypoint(backend: str, **kwargs):
    assert backend == "kernel", f"unsupported backend {backend!r}"
    import flash_mla

    return flash_mla.flash_mla_with_kvcache(**kwargs)
