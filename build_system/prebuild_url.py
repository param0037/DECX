import json
import sys

if __name__ == "__main__":
    args = sys.argv

    package_name = ''
    target_os = ''
    arch = ''
    json_file = ''

    for arg in args:
        arg_lowcase = arg.lower()
        if '-pkg' in arg_lowcase:
            if 'sdl2_image' in arg_lowcase:
                package_name = 'SDL2_image'
                break
            elif 'sdl2' in arg_lowcase:
                package_name = 'SDL2'
                break
            else:
                print('Error while decoding prebuild info: Invalid package name')
                sys.exit()

    for arg in args:
        if '-file' in arg:
            json_file = arg[len('-file='):]

    for arg in args:
        arg_lowcase = arg.lower()
        if '-arch' in arg_lowcase:
            if 'x64' in arg_lowcase:
                arch = 'x64'
                break
            elif 'aarch64' in arg_lowcase or 'arm64' in arg_lowcase:
                arch = 'aarch64'
                break
            else:
                print('Error while decoding prebuild info: Invalid architecture')
                sys.exit()

    for arg in args:
        arg_lowcase = arg.lower()
        if '-os' in arg_lowcase:
            if 'windows' in arg_lowcase:
                target_os = 'Windows'
                break
            elif 'linux' in arg_lowcase:
                target_os = 'Linux'
                break
            else:
                print('Error while decoding prebuild info: Invalid target OS')
                sys.exit()

    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            json_data = json.load(file)

        pkg_info = json_data[package_name]
        pkg_info_os_spec = pkg_info[target_os]
        url = pkg_info_os_spec[arch]
        print(url)

    except FileNotFoundError:
        print('Error while decoding prebuild info: file not found')
        sys.exit()