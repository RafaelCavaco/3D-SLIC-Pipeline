# 3D-SLIC-Pipeline

Texture clustering with geometry-aware SLIC.

Pipeline
--------
1. Load an OBJ mesh and its RGBA texture.
2. Rasterize the OBJ UV triangles into the texture domain to obtain one XYZ
   coordinate for every valid texture texel.
3. Run generalized SLIC on [R, G, B, lambda_xyz*X, lambda_xyz*Y, lambda_xyz*Z]. 
4. Fill each superpixel with its mean RGB value.
5. Convert the mean superpixel colors to CIELAB and optionally scale L*.
6. Run K-means on one mean Lab vector per superpixel.
7. Propagate cluster labels to all texels and save/display the article-style
   pipeline outputs.
8. Optionally select spatially distributed candidate LIBS points per cluster.

The key change from the article is step 3: SLIC uses RGB + mesh XYZ rather than
RGB alone. K-means remains color-based, matching the article.
"""
