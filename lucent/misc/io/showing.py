# Slightly modified from original Lucid library

# Copyright 2018 The Lucid Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Methods for displaying images from Numpy arrays."""

from __future__ import absolute_import, division, print_function

# from io import BytesIO
import base64
from string import Template
import numpy as np
import IPython.display

from lucent.misc.io.serialize_array import serialize_array, array_to_jsbuffer
from lucent.misc.io.collapse_channels import collapse_channels


# create logger with module name, e.g. lucid.misc.io.showing
# log = logging.getLogger(__name__)


def _display_html(html_str):
    IPython.display.display(IPython.display.HTML(html_str))


def _image_url(array, fmt='png', mode="data", quality=90, domain=None):
    """Create a data URL representing an image from a PIL.Image.

    Args:
        image: a numpy array
        mode: presently only supports "data" for data URL

    Returns:
        URL representing image
    """
    supported_modes = ("data")
    if mode not in supported_modes:
        message = "Unsupported mode '%s', should be one of '%s'."
        raise ValueError(message, mode, supported_modes)

    image_data = serialize_array(array, fmt=fmt, quality=quality, domain=domain)
    base64_byte_string = base64.b64encode(image_data).decode('ascii')
    return "data:image/" + fmt.upper() + ";base64," + base64_byte_string


# public functions

def _image_html(array, width=None, domain=None, fmt='png'):
    url = _image_url(array, domain=domain, fmt=fmt)
    style = "image-rendering: pixelated;"
    if width is not None:
        style += "width: {width}px;".format(width=width)
    return """<img src="{url}" style="{style}">""".format(url=url, style=style)


def image(array, domain=None, width=None, fmt='png'):
    """Display an image.

    Args:
        array: NumPy array representing the image
        fmt: Image format e.g. png, jpeg
        domain: Domain of pixel values, inferred from min & max values if None
        width: width of output image, scaled using nearest neighbor interpolation.
            size unchanged if None
    """

    _display_html(
        _image_html(array, width=width, domain=domain, fmt=fmt)
    )


def images(arrays, labels=None, domain=None, width=None):
    """Display a list of images with optional labels.

    Args:
        arrays: A list of NumPy arrays representing images
        labels: A list of strings to label each image.
            Defaults to show index if None
        domain: Domain of pixel values, inferred from min & max values if None
        width: width of output image, scaled using nearest neighbor interpolation.
            size unchanged if None
    """

    string = '<div style="display: flex; flex-direction: row;">'
    for i, array in enumerate(arrays):
        label = labels[i] if labels is not None else i
        img_html = _image_html(array, width=width, domain=domain)
        string += """<div style="margin-right:10px; margin-top: 4px;">
                            {label} <br/>
                            {img_html}
                        </div>""".format(label=label, img_html=img_html)
    string += "</div>"
    _display_html(string)


def show(thing, domain=(0, 1), **kwargs):
    """Display a numpy array without having to specify what it represents.

    This module will attempt to infer how to display your tensor based on its
    rank, shape and dtype. rank 4 tensors will be displayed as image grids, rank
    2 and 3 tensors as images.

    For tensors of rank 3 or 4, the innermost dimension is interpreted as channel.
    Depending on the size of that dimension, different types of images will be
    generated:

        shp[-1]
            = 1  --  Black and white image.
            = 2  --  See >4
            = 3  --  RGB image.
            = 4  --  RGBA image.
            > 4  --  Collapse into an RGB image.
                             If all positive: each dimension gets an evenly spaced hue.
                             If pos and neg: each dimension gets two hues
                                    (180 degrees apart) for positive and negative.

    Common optional arguments:

        domain: range values can be between, for displaying normal images
            None  = infer domain with heuristics
            (a,b) = clip values to be between a (min) and b (max).

        w: width of displayed images
            None  = display 1 pixel per value
            int   = display n pixels per value (often used for small images)

        labels: if displaying multiple objects, label for each object.
            None  = label with index
            []    = no labels
            [...] = label with corresponding list item

    """
    def collapse_if_needed(arr):
        channels = arr.shape[-1]
        if channels not in [1, 3, 4]:
            # log.debug("Collapsing %s channels into 3 RGB channels." % K)
            return collapse_channels(arr)
        return arr

    if isinstance(thing, np.ndarray):
        rank = len(thing.shape)

        if rank in [3, 4]:
            thing = collapse_if_needed(thing)

        if rank == 4:
            # log.debug("Show is assuming rank 4 tensor to be a list of images.")
            images(thing, domain=domain, **kwargs)
        elif rank in (2, 3):
            # log.debug("Show is assuming rank 2 or 3 tensor to be an image.")
            image(thing, domain=domain, **kwargs)
        else:
            # log.warning("Show only supports numpy arrays of rank 2-4. Using repr().")
            print(repr(thing))
    elif isinstance(thing, (list, tuple)):
        # log.debug("Show is assuming list or tuple to be a collection of images.")

        if isinstance(thing[0], np.ndarray) and len(thing[0].shape) == 3:
            thing = [collapse_if_needed(t) for t in thing]

        images(thing, domain=domain, **kwargs)
    else:
        # log.warning("Show only supports numpy arrays so far. Using repr().")
        print(repr(thing))


def textured_mesh(mesh, texture, background='0xffffff'):
    texture_data_url = _image_url(texture, fmt='jpeg', quality=90)

    code = Template('''
    <input id="unfoldBox" type="checkbox" class="control">Unfold</input>
    <input id="shadeBox" type="checkbox" class="control">Shade</input>

    <script src="https://cdn.rawgit.com/mrdoob/three.js/r89/build/three.min.js"></script>
    <script src="https://cdn.rawgit.com/mrdoob/three.js/r89/examples/js/controls/OrbitControls.js"></script>

    <script type="x-shader/x-vertex" id="vertexShader">
        uniform float viewAspect;
        uniform float unfolding_perc;
        uniform float shadeFlag;
        varying vec2 text_coord;
        varying float shading;
        void main () {
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            vec4 plane_position = vec4((uv.x*2.0-1.0)/viewAspect, (uv.y*2.0-1.0), 0, 1);
            gl_Position = mix(gl_Position, plane_position, unfolding_perc);

            //not normalized on purpose to simulate the rotation
            shading = 1.0;
            if (shadeFlag > 0.5) {
                vec3 light_vector = mix(normalize(cameraPosition-position), normal, unfolding_perc);
                shading = dot(normal, light_vector);
            }

            text_coord = uv;
        }
    </script>

    <script type="x-shader/x-fragment" id="fragmentShader">
        uniform float unfolding_perc;
        varying vec2  text_coord;
        varying float shading;
        uniform sampler2D texture;

        void main() {
            gl_FragColor = texture2D(texture, text_coord);
            gl_FragColor.rgb *= shading;
        }
    </script>

    <script>
    "use strict";

    const el = id => document.getElementById(id);

    const unfoldDuration = 1000.0;
    var camera, scene, renderer, controls, material;
    var unfolded = false;
    var unfoldStart = -unfoldDuration*10.0;

    init();
    animate(0.0);

    function init() {
        var width = 800, height = 600;

        scene = new THREE.Scene();

        camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 100);
        camera.position.z = 3.3;
        scene.add(camera);

        controls = new THREE.OrbitControls( camera );

        var geometry = new THREE.BufferGeometry();
        geometry.addAttribute( 'position', new THREE.BufferAttribute($verts, 3 ) );
        geometry.addAttribute( 'uv', new THREE.BufferAttribute($uvs, 2) );
        geometry.setIndex(new THREE.BufferAttribute($faces, 1 ));
        geometry.computeVertexNormals();

        var texture = new THREE.TextureLoader().load('$tex_data_url', update);
        material = new THREE.ShaderMaterial( {
            uniforms: {
                viewAspect: {value: width/height},
                unfolding_perc: { value: 0.0 },
                shadeFlag: { value: 0.0 },
                texture: { type: 't', value: texture },
            },
            side: THREE.DoubleSide,
            vertexShader: el( 'vertexShader' ).textContent,
            fragmentShader: el( 'fragmentShader' ).textContent
        });

        var mesh = new THREE.Mesh(geometry, material);
        scene.add(mesh);
        scene.background = new THREE.Color( $background );

        renderer = new THREE.WebGLRenderer({antialias: true});
        renderer.setSize(width, height);
        document.body.appendChild(renderer.domElement);

        // render on change only
        controls.addEventListener('change', function() {
            // fold mesh back if user wants to interact
            el('unfoldBox').checked = false;
            update();
        });
        document.querySelectorAll('.control').forEach(e=>{
            e.addEventListener('change', update);
        });
    }

    function update() {
        requestAnimationFrame(animate);
    }

    function ease(x) {
        x = Math.min(Math.max(x, 0.0), 1.0);
        return x*x*(3.0 - 2.0*x);
    }

    function animate(time) {
        var unfoldFlag = el('unfoldBox').checked;
        if (unfolded != unfoldFlag) {
            unfolded = unfoldFlag;
            unfoldStart = time - Math.max(unfoldStart+unfoldDuration-time, 0.0);
        }
        var unfoldTime = (time-unfoldStart) / unfoldDuration;
        if (unfoldTime < 1.0) {
            update();
        }
        var unfoldVal = ease(unfoldTime);
        unfoldVal = unfolded ? unfoldVal : 1.0 - unfoldVal;
        material.uniforms.unfolding_perc.value = unfoldVal;

        material.uniforms.shadeFlag.value = el('shadeBox').checked ? 1.0 : 0.0;
        controls.update();
        renderer.render(scene, camera);
    }
    </script>
    ''').substitute(
        verts=array_to_jsbuffer(mesh['position'].ravel()),
        uvs=array_to_jsbuffer(mesh['uv'].ravel()),
        faces=array_to_jsbuffer(np.uint32(mesh['face'].ravel())),
        tex_data_url=texture_data_url,
        background=background,
    )
    _display_html(code)


def animate_sequence(sequence, domain=(0, 1), fmt='png'):
    steps, height, width, _ = sequence.shape
    sequence = np.concatenate(sequence, 1)
    code = Template('''
    <style> 
        #animation {
            width: ${width}px;
            height: ${height}px;
            background: url('$image_url') left center;
            animation: play 1s steps($steps) infinite alternate;
        }
        @keyframes play {
            100% { background-position: -${sequence_width}px; }
        }
    </style><div id='animation'></div>
    ''').substitute(
        image_url=_image_url(sequence, domain=domain, fmt=fmt),
        sequence_width=width*steps,
        width=width,
        height=height,
        steps=steps,
    )
    _display_html(code)
