from src.app.session import Session
from src.federated.subscribers.wandb_logger import WandbLogger


class SessionWandbLogger(WandbLogger):
    def __init__(self, session: Session):
        super().__init__(session.settings.configs, resume=True, id=session.session_id())
