from __future__ import annotations

import time

import structlog
from asgi_correlation_id.context import correlation_id
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp
from starlette.types import Message
from starlette.types import Receive
from starlette.types import Scope
from starlette.types import Send
from structlog.stdlib import BoundLogger
from uvicorn.protocols.utils import get_path_with_query_string


def truncate_body(body: bytes) -> bytes:
    """Truncate body when logging to avoid stressing path operations

    Args:
        body (bytes): request body

    Returns:
        bytes: truncated request body
    """

    def convert_bytes_unit(size: int) -> str:
        if size >= 1024 * 1024 * 1024:
            return f'{size / (1024 * 1024 * 1024):.2f} GB'
        elif size >= 1024 * 1024:
            return f'{size / (1024 * 1024):.2f} MB'
        elif size >= 1024:
            return f'{size / 1024:.2f} KB'
        else:
            return f'{size} bytes'

    if len(body) > 100:
        return body[:100] + f'... Truncated {convert_bytes_unit(len(body))}'.encode()
    return body


class LoggingMiddleware:
    def __init__(self, app: ASGIApp, logger: BoundLogger) -> None:
        self.app = app
        self.logger = logger

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return

        structlog.contextvars.clear_contextvars()
        # These context vars will be added to all log entries emitted during the request
        request_id = correlation_id.get()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        start_time = time.perf_counter_ns()

        client_host = None
        client_port = None
        if 'client' in scope:
            client_host = scope['client'][0]
            client_port = scope['client'][1]
        url = get_path_with_query_string(scope)  # type: ignore
        http_method = scope['method']
        http_version = scope['http_version']
        body = b''
        status_code = 500
        process_time = 0

        async def receive_logging():
            nonlocal body
            message = await receive()
            assert message['type'] == 'http.request'
            body = truncate_body(message['body'])
            return message

        async def send_logging(message: Message) -> None:
            nonlocal start_time
            nonlocal process_time
            nonlocal status_code
            process_time = time.perf_counter_ns() - start_time
            if message['type'] == 'http.response.start':
                status_code = message['status']
                headers = MutableHeaders(scope=message)
                headers.append(key='X-Process-Time', value=str(process_time / 10**9))
            await send(message)

        try:
            await self.app(scope, receive_logging, send_logging)
        except Exception:
            structlog.stdlib.get_logger('api.error').exception('Uncaught exception')
            raise
        finally:
            self.logger.info(
                f"""{client_host}:{client_port} - "{http_method} {url} HTTP/{http_version}" {status_code}""",
                http={
                    'url': url,
                    'status_code': status_code,
                    'method': http_method,
                    'request_id': request_id,
                    'version': http_version,
                    'body': body,
                },
                duration=process_time,
            )
