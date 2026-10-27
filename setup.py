from setuptools import find_packages, setup

package_name = "mppi"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    package_data={
        package_name: [
            "Dynamics/*.cu",
            "Dynamics/*.cpp",
            "cuda/*.cu",
            "Configs/*.yaml",
        ],
    },
    include_package_data=True,
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "torch"],
    zip_safe=False,
    maintainer="hound",
    maintainer_email="hound@todo.todo",
    description="Standalone MPPI (MPPI + sampling + CUDA bicycle + example costs).",
    license="MIT",
)
