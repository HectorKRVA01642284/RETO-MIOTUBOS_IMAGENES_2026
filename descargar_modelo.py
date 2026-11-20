"""Descarga y verifica el único checkpoint provisional publicado."""
from hashlib import sha256
import json
from pathlib import Path
from urllib.request import urlopen


def main():
    root = Path(__file__).resolve().parent
    profile = json.loads((root / 'configuracion/modelo-provisional.json').read_text(encoding='utf8'))
    destination = root / 'modelos' / profile['archivo']
    destination.parent.mkdir(exist_ok=True)
    if destination.exists():
        if sha256(destination.read_bytes()).hexdigest() != profile['sha256']:
            raise ValueError('El archivo existente no coincide con el perfil; se conserva para revisión.')
        print(f'Modelo existente verificado: {destination}')
        return
    temporary = destination.with_suffix('.pt.partial')
    digest = sha256()
    try:
        with urlopen(profile['descarga'], timeout=180) as source, temporary.open('wb') as output:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
        if temporary.stat().st_size != profile['bytes'] or digest.hexdigest() != profile['sha256']:
            raise ValueError('La descarga no coincide con el tamaño o SHA-256 del perfil.')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Modelo descargado y verificado: {destination}')


if __name__ == '__main__':
    main()
