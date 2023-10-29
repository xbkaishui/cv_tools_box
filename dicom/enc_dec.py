import os
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import traceback

# 解密文件
def decrypt_file(file_path, key):
    with open(file_path, 'rb') as encrypted_file:
        nonce = encrypted_file.read(16)
        ciphertext = encrypted_file.read()
        # tag = encrypted_file.read(16)
        # print(f'decr mac {tag}')
        cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
        plaintext = cipher.decrypt(ciphertext)
        try:
            # cipher.verify(tag)
            with open(file_path, 'wb') as decrypted_file:
                decrypted_file.write(plaintext)
            print(f'Decrypted: {file_path}')
        except ValueError:
            traceback.print_exc()
            print(f'Failed to decrypt {file_path}. The file may be corrupted or the key is incorrect.')

# 生成一个随机密钥
def generate_key():
    return get_random_bytes(16)

# 加密文件
def encrypt_file(file_path, key):
    cipher = AES.new(key, AES.MODE_EAX)
    with open(file_path, 'rb') as file:
        plaintext = file.read()
        ciphertext, tag = cipher.encrypt_and_digest(plaintext)
        with open(file_path, 'wb') as encrypted_file:
            encrypted_file.write(cipher.nonce)
            encrypted_file.write(ciphertext)
            # print(f'mac {tag}')
            # encrypted_file.write(tag)

# 遍历目录并加密所有文件
def encrypt_directory(directory, key):
    for root, _, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            encrypt_file(file_path, key)
            print(f'Encrypted: {file_path}')
            
def decrypt_directory(directory_to_decrypt, key):
    # 调用函数解密目录中的所有文件
    for root, _, files in os.walk(directory_to_decrypt):
        for filename in files:
            file_path = os.path.join(root, filename)
            decrypt_file(file_path, key)

if __name__ == "__main__":
    # 设置要加密的目录和密钥
    directory_to_encrypt = '/tmp/bb'
    key = generate_key()
    print(key)

    # 调用函数加密目录中的所有文件
    encrypt_directory(directory_to_encrypt, key)
    
    # # 设置要解密的目录和密钥
    # directory_to_decrypt = '/path/to/your/encrypted/directory'
    # key = b'your_secret_key_here'  # 用加密时使用的密钥替换此处的密钥
    print(key)
    decrypt_directory(directory_to_encrypt, key)