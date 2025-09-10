# src/utils/pydantic_bridge.py
def apply_pydantic_v1_bridge():
    """
    Make Pydantic v2 behave like v1 for legacy code:
      - BaseModel.dict() -> model_dump()
      - Add BOTH legacy keys in the returned dict: '__fields_set__' and 'fields_set'
      - BaseModel.json() -> model_dump_json()
      - Expose read-only properties: __fields_set__ and fields_set
    This prevents KeyError on "del d['fields_set']" or "del d['__fields_set__']".
    """
    try:
        import pydantic as _pd
        from pydantic import BaseModel as _BM

        # Only patch if v2 is actually installed
        if int((_pd.__version__.split('.', 1)[0]) or "2") < 2:
            return

        _dump = _BM.model_dump

        def _dict_v1(self, *args, **kwargs):
            d = _dump(self, *args, **kwargs)
            fs = (
                getattr(self, "__pydantic_fields_set__", None)
                or getattr(self, "model_fields_set", None)
                or set()
            )
            try:
                legacy = set(fs)
            except Exception:
                legacy = set()

            # Inject BOTH keys so any old code deleting either won't crash
            d.setdefault("__fields_set__", legacy)
            d.setdefault("fields_set", legacy)
            return d

        # v1-like methods
        _BM.dict = _dict_v1
        _BM.json = _BM.model_dump_json

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

        # Read-only properties (no setter!)
        try:
            setattr(_BM, "__fields_set__", property(_get_fields_set))
        except Exception:
            pass
        try:
            setattr(_BM, "fields_set", property(_get_fields_set))
        except Exception:
            pass

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"pydantic v1 bridge not applied: {e}")
