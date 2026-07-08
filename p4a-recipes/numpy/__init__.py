# numpy の Android ビルドを直すための差し替えレシピ
#
# python-for-android 標準の numpy レシピは v2.3.0 を使うが、
# このバージョンには unique.cpp に #include <unordered_map> が
# 抜けているバグがあり、NDK の libc++ ではコンパイルに失敗する。
# (upstream の修正: https://github.com/numpy/numpy/pull/29662)
#
# ここでは標準レシピと同じ内容に、その1行を足すパッチだけを加える。
# buildozer.spec の p4a.local_recipes がこのフォルダを指しているので、
# 標準レシピの代わりにこちらが使われる。

from os.path import join
import shutil

from pythonforandroid.recipe import Recipe, MesonRecipe


class NumpyRecipe(MesonRecipe):
    version = "v2.3.0"
    url = "git+https://github.com/numpy/numpy"
    extra_build_args = ["-Csetup-args=-Dblas=none", "-Csetup-args=-Dlapack=none"]
    opt_depends = ["libopenblas"]
    need_stl_shared = True
    min_ndk_api_support = 24

    # ここが標準レシピとの唯一の違い (includeが抜けているのを直す)
    patches = ["unique-unordered-map.patch"]

    def get_include(self, arch):
        return join(
            self.ctx.get_python_install_dir(arch.arch), "numpy/_core/include",
        )

    def get_recipe_meson_options(self, arch):
        options = super().get_recipe_meson_options(arch)
        options["properties"]["longdouble_format"] = (
            "IEEE_DOUBLE_LE" if arch.arch in ["armeabi-v7a", "x86"] else "IEEE_QUAD_LE"
        )
        return options

    def get_recipe_env(self, arch, **kwargs):
        env = super().get_recipe_env(arch, **kwargs)
        # _PYTHON_HOST_PLATFORM declares that we're cross-compiling
        # and avoids issues when building on macOS for Android targets.
        env["_PYTHON_HOST_PLATFORM"] = arch.command_prefix
        # NPY_DISABLE_SVML=1 allows numpy to build for non-AVX512 CPUs
        # See: https://github.com/numpy/numpy/issues/21196
        env["NPY_DISABLE_SVML"] = "1"
        env["TARGET_PYTHON_EXE"] = join(
            Recipe.get_recipe("python3", self.ctx).get_build_dir(arch.arch),
            "android-build",
            "python",
        )
        blas_dir = join(Recipe.get_recipe("libopenblas", self.ctx
                                          ).get_build_dir(arch.arch), "build")
        blas_incdir = blas_dir
        blas_libdir = join(blas_dir, "lib")
        env["CXXFLAGS"] += f" -I{blas_incdir} -L{blas_libdir}"
        if 'libopenblas' in self.ctx.recipe_build_order:
            self.extra_build_args = [
                "-Csetup-args=-Dblas=auto",
                "-Csetup-args=-Dlapack=auto",
                "-Csetup-args=-Dallow-noblas=False",
            ]
        return env

    def get_hostrecipe_env(self, arch=None):
        env = super().get_hostrecipe_env(arch=arch)
        env["RANLIB"] = shutil.which("ranlib")
        return env


recipe = NumpyRecipe()
