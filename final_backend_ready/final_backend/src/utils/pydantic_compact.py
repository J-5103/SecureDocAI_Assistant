def apply_pydantic_compat():
    """
    Pydantic v2 compat for legacy v1-style code:
    - read-only __fields_set__ property (maps to v2 field set)
    - alias .dict() / .json() to model_dump() / model_dump_json()
    """
    try:
        import pydantic as _pd
        from pydantic import BaseModel as _BM
        major = int((_pd.__version__.split(".", 1)[0]) or "2")
        if major < 2:
            return

        def _get_fields_set(self):
            fs = (
                getattr(self, "__pydantic_fields_set__", None)
                or getattr(self, "model_fields_set", None)
                or set()
            )
            try:
                return set(fs)
            except Exception:
                return set()

        # expose read-only property; DO NOT set per-instance
        try:
            setattr(_BM, "__fields_set__", property(_get_fields_set))
        except Exception:
            pass

        if not hasattr(_BM, "dict"):
            _BM.dict = _BM.model_dump  # type: ignore[attr-defined]
        if not hasattr(_BM, "json"):
            _BM.json = _BM.model_dump_json  # type: ignore[attr-defined]
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Pydantic compat shim not applied: {e}")
