"""API module for FastAPI"""
from typing import Callable, Dict, Optional
from threading import Lock
from secrets import compare_digest

from modules import shared  # pylint: disable=import-error
from modules.api.api import decode_base64_to_image  # pylint: disable=E0401
from modules.call_queue import queue_lock  # pylint: disable=import-error
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from tagger import utils  # pylint: disable=import-error
from tagger import api_models as models  # pylint: disable=import-error


class Api:
    """Api class for FastAPI"""
    def __init__(
        self, app: FastAPI, qlock: Lock, prefix: Optional[str] = None
    ) -> None:
        if shared.cmd_opts.api_auth:
            self.credentials = {}
            for auth in shared.cmd_opts.api_auth.split(","):
                user, password = auth.split(":")
                self.credentials[user] = password

        self.app = app
        self.queue_lock = qlock
        self.prefix = prefix
        self.images: Dict[str, object] = {}

        self.add_api_route(
            'interrogate',
            self.endpoint_interrogate,
            methods=['POST'],
            response_model=models.TaggerInterrogateResponse
        )

        self.add_api_route(
            'interrogators',
            self.endpoint_interrogators,
            methods=['GET'],
            response_model=models.TaggerInterrogatorsResponse
        )

        self.add_api_route(
            'unload-interrogators',
            self.endpoint_unload_interrogators,
            methods=['POST'],
            response_model=str,
        )
        self.add_api_route(
            'queue-image',
            self.endpoint_queue_image,
            methods=['POST'],
            response_model=models.TaggerQueueImageResponse
        )
        self.add_api_route(
            'batch-process',
            self.endpoint_batch,
            methods=['POST'],
            response_model=models.TaggerBatchResponse
        )

    def auth(self, creds: Optional[HTTPBasicCredentials] = None):
        if creds is None:
            creds = Depends(HTTPBasic())
        if creds.username in self.credentials:
            if compare_digest(creds.password,
                              self.credentials[creds.username]):
                return True

        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={
                "WWW-Authenticate": "Basic"
            })

    def add_api_route(self, path: str, endpoint: Callable, **kwargs):
        if self.prefix:
            path = f'{self.prefix}/{path}'

        if shared.cmd_opts.api_auth:
            return self.app.add_api_route(path, endpoint, dependencies=[
                   Depends(self.auth)], **kwargs)
        return self.app.add_api_route(path, endpoint, **kwargs)

    def endpoint_interrogate(self, req: models.TaggerInterrogateRequest):
        """ one file interrogation """
        if req.image is None:
            raise HTTPException(404, 'Image not found')

        if req.model not in utils.interrogators:
            raise HTTPException(404, 'Model not found')

        image = decode_base64_to_image(req.image)
        with self.queue_lock:
            interrogator = utils.interrogators[req.model]
            rating, tag = interrogator.interrogate(image)
            if req.threshold > 0.0:
                tag = {k: v for k, v in tag.items() if v > req.threshold}
            res = {**rating, **tag}

        return models.TaggerInterrogateResponse(res)

    def endpoint_queue_image(self, req: models.TaggerInterrogateRequest):
        """ post image to queue """
        if req.image is None:
            raise HTTPException(404, 'Image not found')

        # TODO make this a command line option
        if len(self.images) >= getattr(shared.cmd_opts, 'queue_size', 512):
            raise HTTPException(429, 'Queue is full')

        # clobber existing image
        if req.name in self.images:
            i = 0
            while f'{req.name}.{i}' in self.images:
                i = i + 1
            req.name = f'{req.name}.{i}'

        self.images[req.name] = decode_base64_to_image(req.image)

        return models.TaggerQueueImageResponse(True)

    def endpoint_batch(self, req: models.TaggerBatchRequest):
        """ batch interrogation """
        if req.image is None:
            raise HTTPException(404, 'Image not found')

        if req.model not in utils.interrogators:
            raise HTTPException(404, 'Model not found')

        res = {}

        with self.queue_lock:
            interrogator = utils.interrogators[req.model]
            for name, i in self.images.items():
                rating, tag = interrogator.interrogate(i)
                if req.threshold > 0.0:
                    tag = {k: v for k, v in tag.items() if v > req.threshold}
                res[name] = {**rating, **tag}

            # last image
            image = decode_base64_to_image(req.image)
            res[req.name] = interrogator.interrogate(image)

        self.images.clear()

        return models.TaggerBatchResponse(res)

    def endpoint_interrogators(self):
        return models.TaggerInterrogatorsResponse(
            models=list(utils.interrogators.keys())
        )

    def endpoint_unload_interrogators(self):
        unloaded_models = 0

        for i in utils.interrogators.values():
            if i.unload():
                unloaded_models = unloaded_models + 1

        return f"Successfully unload {unloaded_models} model(s)"


def on_app_started(_, app: FastAPI):
    Api(app, queue_lock, '/tagger/v1')
