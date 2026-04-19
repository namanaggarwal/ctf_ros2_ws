from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ctf_game_server'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hamilton',
    maintainer_email='seras@mit.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'game_server = ctf_game_server.game_server:main',
            'publish_dummy_graph = ctf_game_server.publish_dummy_graph:main',
            'publish_obs = ctf_game_server.publish_obs:main',
            'publish_boxes = ctf_game_server.publish_boxes:main'
        ],
    },
)