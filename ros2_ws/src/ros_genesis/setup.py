from setuptools import find_packages, setup

package_name = 'ros_genesis'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools' 'genesis'],
    zip_safe=True,
    maintainer='jachinshen',
    maintainer_email='jachinshen@foxmail.com',
    description='Integrate Genesis',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simulator = ros_genesis.simulator:main'
        ],
    },
)
