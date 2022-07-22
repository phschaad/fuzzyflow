# FuzzyFlow

Verify transformations using:

`python -m fuzzyflow.cli.xfuzz -p <PATH_TO_SDFG> -x <PATH_TO_TRANSFORMATION_JSON>`

Transformation JSON files can be obtained through the transformation's `.to_json()` method or the DaCe serialization module, or through VSCode:

![image](https://user-images.githubusercontent.com/9193712/180575679-3e5a61c0-9c2d-4332-b377-2f043de9cfa3.png)

