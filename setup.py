# Copyright 2019, zhoudoao@gmail.com.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup
from setuptools import find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()


setup(
    name='bayes_vae',
    version='0.0.1',
    description='An package of Bayesian vae',
    long_description=readme,
    author='zhoudoao@gmail.com',
    url='https://github.com/zhoudoao/bayes_vae',
    license=license,
    packages=find_packages(exclude=('testing'))
)
