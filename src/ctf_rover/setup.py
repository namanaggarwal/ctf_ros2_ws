from setuptools import find_packages, setup

package_name = 'ctf_rover'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', [
            'config/params_blue_0.yaml',
            'config/params_blue_1.yaml',
            'config/params_red_0.yaml',
            'config/params_red_1.yaml',
        ]),
        ('share/' + package_name + '/policies', [
            'policies/blue_mappo_final.zip',
            'policies/iter3_red_br.zip',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hamilton',
    maintainer_email='seras@mit.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'rover_node = ctf_rover.rover_node:main',
            'global_to_local_goal = ctf_rover.global_to_local_goal:main',
            'publish_global_goal = ctf_rover.publish_global_goal:main',
            'publish_real_graph = ctf_rover.publish_real_graph:main'
        ],
    },
)
