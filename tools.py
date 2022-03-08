from Crypto.Cipher import AES
from binascii import b2a_hex, a2b_hex

key = "jhuy"
mode = AES.MODE_CBC


def _key_padding_(key: str) -> str:
    """
    填充密钥
    :param key: 密钥
    :return: 填充后的密钥
    """
    key_len = len(key)
    if key_len <= 16:
        for _ in range(key_len, 16):
            key += '\0'
        return key
    if 16 < key_len <= 24:
        for _ in range(key_len, 24):
            key += '\0'
        return key
    if 24 < key_len <= 32:
        for _ in range(key_len, 32):
            key += '\0'
        return key
    return key[:32]


def _text_padding_(text: str) -> str:
    """
    明文填充为16的倍数
    :param text: 明文
    :return: 填充后的明文
    """
    while len(text) % 16 != 0:
        text += '\0'
    return text


def _encrypt_(data: bytes, password: bytes) -> str:
    """
    加密
    :param data: 明文
    :param password: 密钥
    :return:
    """
    cryptor = AES.new(password, mode, password)
    cipher = cryptor.encrypt(data)
    return b2a_hex(cipher).decode('utf-8')


def do_encrypt(data: str) -> str:
    """
    执行加密
    :param data:明文
    :return: 密文
    """
    password = _key_padding_(key).encode('utf-8')
    text = _text_padding_(data).encode('utf-8')
    return _encrypt_(text, password)


def _decrypt_(data: bytes, password: bytes) -> str:
    """
    解密
    :param data: 密文
    :param password: 密钥
    :return: 明文
    """
    cryptor = AES.new(password, mode, password)
    plain_text = cryptor.decrypt(a2b_hex(data))
    # 消除添加的空
    return plain_text.decode('utf-8').rstrip('\0')


def do_decrypt(data: str) -> str:
    """
    执行解密
    :param data:密文
    :return: 明文
    """
    password = _key_padding_(key).encode('utf-8')
    data = data.encode('utf-8')
    return _decrypt_(data, password)
