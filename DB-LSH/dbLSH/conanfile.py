from conan import ConanFile
from conan.tools.cmake import CMakeDeps
from conan.tools.cmake import CMakeToolchain



generators = ("CMakeToolchain", "CMakeDeps")

class Discotec (ConanFile):
    settings = ("os", "build_type", "arch", "compiler")

    def requirements(self):
        self.requires("boost/1.73.0")

    def build_requirements(self):
        self.tool_requires("cmake/[>=2.8]")
        
    def generate(self):
        deps = CMakeToolchain(self)
        deps.check_components_exist = True
        deps.generate()


    def layout(self):
        self.folders.generators = ""
