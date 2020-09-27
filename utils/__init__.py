
from os import path


package_path = path.dirname(__file__)
sub_package_names = ['common', 'data', 'model']

__path__.extend(path.join(package_path, sub_package_name) for sub_package_name in sub_package_names)
