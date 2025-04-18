# Material Generator from Photos

PBR texture generator tool from photo image using AI.

THis tool is based on the [ArmorLab](https://github.com/armory3d/armortools/tree/g4/armorlab) project, which is a set of tools for the Armory3D game engine.

## example usage


First, you need to install onnx files from [Here](https://github.com/armory3d/armorai/releases) to `data` directory.

```bash

git clone https://github.com/20niship/texture2pbr-generator.git
cd texture2pbr-generator

# setup python environment using uv
uv install
uv run main.py
```

Then, you can show normal, roughness, ambient occlusion textures generated from photo image.


## references

- https://github.com/armory3d/armortools/blob/g4/armorlab/sources/nodes/photo_to_pbr_node.ts

