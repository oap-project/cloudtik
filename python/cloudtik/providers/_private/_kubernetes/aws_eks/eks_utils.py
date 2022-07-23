import socket
from OpenSSL import SSL
from OpenSSL.SSL import (VERIFY_PEER)
from OpenSSL.crypto import load_certificate, FILETYPE_PEM

from cloudtik.core._private.cli_logger import cli_logger


def get_client_cert_cas(hostname, port):
    certs = []

    def verify_peer_callback(conn, cert, errnum, depth, ok):
        if cert:
            cli_logger.verbose("Certificate Begin--------------------------------------")
            cli_logger.verbose("    depth = {}, subject = {}", depth, cert.get_subject())
            cli_logger.verbose("    issuer = {}, digest = {}", cert.get_issuer(), cert.digest("sha1"))
            cli_logger.verbose("Certificate End----------------------------------------")
            cert_info = {
                "depth": depth,
                "subject": cert.get_subject(),
                "issuer": cert.get_issuer(),
                "digest": cert.digest(),
            }
            certs.append(cert_info)
        return 1
    ctx = SSL.Context(SSL.SSLv23_METHOD)
    # set_default_verify_paths causes the platform-specific CA certificate locations
    # to be used for verification purposes
    # ctx.set_default_verify_paths()
    ctx.set_verify(VERIFY_PEER, verify_peer_callback)
    sock = SSL.Connection(ctx, socket.socket(socket.AF_INET, socket.SOCK_STREAM))
    sock.connect((hostname, port))
    sock.set_connect_state()
    sock.set_tlsext_host_name(hostname.encode("utf-8"))
    sock.do_handshake()
    return certs


def get_sha1_digest(x509_certificate_str):
    cert = load_certificate(FILETYPE_PEM, x509_certificate_str)
    sha1_digest = cert.digest("sha1")
    return sha1_digest


def get_root_ca_cert(hostname, port):
    certs = get_client_cert_cas(hostname, port)
    count = len(certs)
    if count == 0:
        raise RuntimeError("Failed to retrieve CA certs from server.")

    def sort_by_depth(cert_info):
        return cert_info["depth"]
    certs.sort(key=sort_by_depth)
    return certs[count - 1]


def get_root_ca_cert_thumbprint(hostname, port):
    cert_info = get_root_ca_cert(hostname, port)
    digest = cert_info["digest"]
    return digest.decode("utf-8").replace(":", "").lower()
