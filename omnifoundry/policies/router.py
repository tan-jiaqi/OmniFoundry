"""ModelRouter: role name -> Client. Roles and their endpoints are config data (`models:`).

`models.defaults` is merged into every role; a role may `inherit: <other role>` to share settings.
Swapping the generator or verifier for a self-trained model is a config change.
"""
from ..config import deep_merge
from ..llm import Client


class ModelRouter:
    def __init__(self, models, cache, ledger=None):
        self.spec = models or {}
        self.cache = cache
        self.ledger = ledger
        self._clients = {}

    def options(self, role, _seen=()):
        if role not in self.spec:
            raise KeyError(f"no model configured for role {role!r}; add it under `models:`")
        own = dict(self.spec[role])
        parent = own.pop("inherit", None)
        if parent:
            if parent in _seen:
                raise ValueError(f"model role inheritance loop at {role}")
            base = self.options(parent, _seen + (role,))
        else:
            base = self.spec.get("defaults", {})
        return deep_merge(base, own)

    def __call__(self, role):
        if role not in self._clients:
            self._clients[role] = Client(role, self.options(role), self.cache, self.ledger)
        return self._clients[role]

    def model_ids(self, *roles):
        return {r: self.options(r).get("model", "") for r in roles}
