import pytest

from cloudtik.core._private.crypto import AESCipher

TEST_TEXTS = ['abc', 'In Paris of the nineteenth century',
              'poor Jean Valjean was Convicted for stealing a loaf of bread t and sent to prison for five years',
              'At last he was paroled from prison nineteen years later. Rejected by society for being a former convict']


class TestCrypto:
    def test_generate_key(self):
        secrets = AESCipher.generate_key()
        assert len(secrets) == 32

    @pytest.mark.parametrize("plain_text", TEST_TEXTS)
    def test_encrypt_decrypt(self, plain_text):
        # Encrypt and put
        secrets = AESCipher.generate_key()
        cipher = AESCipher(secrets)

        encrypted_str = cipher.encrypt(plain_text)
        d_plain_text = cipher.decrypt(encrypted_str)
        assert d_plain_text == plain_text

        cipher_de = AESCipher(secrets)
        d_plain_text_1 = cipher_de.decrypt(encrypted_str)
        assert d_plain_text_1 == plain_text


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
