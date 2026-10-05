#!/usr/bin/env python3
"""Run the posed STL packager without forwarding GitHub auth to signed storage URLs."""
from __future__ import annotations

from urllib.parse import urlsplit
import urllib.request


class StripSensitiveHeadersOnCrossHostRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None
        old_host = urlsplit(req.full_url).netloc.casefold()
        new_host = urlsplit(newurl).netloc.casefold()
        if old_host != new_host:
            for header in (
                "Authorization",
                "X-GitHub-Api-Version",
                "Accept",
            ):
                redirected.remove_header(header)
        return redirected


urllib.request.install_opener(
    urllib.request.build_opener(StripSensitiveHeadersOnCrossHostRedirect())
)

from package_available_photo_posed_stls import main


if __name__ == "__main__":
    raise SystemExit(main())
