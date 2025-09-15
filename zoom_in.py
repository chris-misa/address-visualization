"""
Script to render zoom-in on Hilbert-space animation
"""

import math
import os
import sys

import glm
import moderngl
import numpy as np
import pandas as pd
import pygame
from hilbertcurve.hilbertcurve import HilbertCurve
import ipaddress

COLOR_SCALE = 0.75

class TextTex:
    def __init__(self, init_text, font_size = 32):
        self.ctx = moderngl.get_context()
        self.font = pygame.font.Font(None, font_size)
        self.update_text(init_text)

    def update_text(self, text):
        text = self.font.render(text, True, (255,255,255))
        self.tex = self.ctx.texture(text.get_size(), 4, pygame.image.tobytes(text, "RGBA"))
        self.sampler = self.ctx.sampler(texture = self.tex)
        self.sampler.filter = (self.ctx.NEAREST, self.ctx.NEAREST)

    def use(self):
        self.sampler.use()

# From: https://github.com/moderngl/moderngl/blob/main/examples/08_texture.py
class PlaneGeometry:
    def __init__(self):
        self.ctx = moderngl.get_context()
        vertices = np.array([
            -0.5, 0.5, 0.0, 0.0, 0.0,
            0.5, 0.5, 0.0, 1.0, 0.0,
            -0.5, -0.5, 0.0, 0.0, 1.0,
            -0.5, -0.5, 0.0, 0.0, 1.0,
            0.5, 0.5, 0.0, 1.0, 0.0,
            0.5, -0.5, 0.0, 1.0, 1.0,
        ])

        self.vbo = self.ctx.buffer(vertices.astype('f4').tobytes())

    def vertex_array(self, program):
        return self.ctx.vertex_array(program, [(self.vbo, '3f 2f', 'in_vertex', 'in_uv')])
        

class TextPlane:
    def __init__(self, init_text, font_size = 32, x = 0.0, y = 1.0):
        self.ctx = moderngl.get_context()
        self.prog = self.ctx.program(
            vertex_shader = '''
            #version 330 core
            uniform float scale = 1.0;
            uniform float dx = 0.0;
            uniform float dy = 0.0;
            layout (location = 0) in vec3 in_vertex;
            layout (location = 1) in vec2 in_uv;
            out vec2 v_uv;

            void main() {
              v_uv = in_uv;
              gl_Position = vec4(in_vertex * scale, 1.0);
              gl_Position.x -= dx;
              gl_Position.y -= dy;
            }
            ''',
            fragment_shader = '''
            #version 330 core
            uniform sampler2D Tex;
            in vec2 v_uv;
            void main() {
              gl_FragColor = texture(Tex, v_uv);
            }
            '''
        )
        _, _, sw, sh = self.ctx.viewport
        self.prog['scale'] = font_size / sh
        self.prog['dy'] = y - (font_size / sh)
        self.prog['dx'] = x
        self.tex = TextTex(init_text, font_size = font_size)
        self.geom = PlaneGeometry().vertex_array(self.prog)

    def update_text(self, text):
        self.tex.update_text(text)

    def render(self):
        self.tex.use()
        self.geom.render()

class ColorScale:
    def __init__(self):
        self.ctx = moderngl.get_context()
        self.prog = self.ctx.program(
            vertex_shader = '''
            #version 330 core
            uniform float hscale = 1.0;
            uniform float vscale = 1.0;
            uniform float dy = 0.0;
            layout (location = 0) in vec3 in_vertex;
            layout (location = 1) in vec2 in_uv;
            out vec2 v_uv;

            void main() {
              v_uv = in_uv;
              vec3 v = in_vertex;
              v.x *= hscale;
              v.y *= vscale;
              v.y += dy;
              gl_Position = vec4(v, 1.0);
            }
            ''',
            fragment_shader = f'''
            #version 330 core
            in vec2 v_uv;
            
            //  Function from Iñigo Quiles
            //  https://www.shadertoy.com/view/MsS3Wc
            vec3 hsb2rgb( in vec3 c ) {{
                vec3 rgb = clamp(abs(mod(c.x*6.0+vec3(0.0,4.0,2.0),
                                         6.0)-3.0)-1.0,
                                 0.0,
                                 1.0 );
                rgb = rgb*rgb*(3.0-2.0*rgb);
                return c.z * mix(vec3(1.0), rgb, c.y);
            }}

            void main() {{
              vec3 c = hsb2rgb(vec3(1.0 - {COLOR_SCALE} * v_uv.x, 1.0, 1.0));
              gl_FragColor = vec4(c, 1.0);
            }}
            '''
        )
        _, _, sw, sh = self.ctx.viewport
        self.prog['hscale'] = 400 / sw
        self.prog['vscale'] = 50 / sh
        self.prog['dy'] = 1 - (50 / sh)
        self.geom = PlaneGeometry().vertex_array(self.prog)

    def render(self):
        self.geom.render()


class LegendKey:
    def __init__(self, hsize, vsize):
        self.ctx = moderngl.get_context()
        vertices = np.array([
            -0.5, -0.5, 0.0,
            -0.5, 0.5, 0.0,
            -0.5, 0.0, 0.0,
            0.5, 0.0, 0.0,
            0.5, -0.5, 0.0,
            0.5, 0.5, 0.0
        ])
        self.vbo = self.ctx.buffer(vertices.astype('f4').tobytes())
        self.prog = self.ctx.program(
            vertex_shader = '''
            #version 330 core
            uniform float hscale = 1.0;
            uniform float vscale = 1.0;
            uniform float dy = 0.0;
            layout (location = 0) in vec3 in_vertex;

            void main() {
              vec3 v = in_vertex;
              v.x *= hscale;
              v.y *= vscale;
              gl_Position = vec4(v, 1.0);
              gl_Position.y += dy;
            }
            ''',
            fragment_shader = '''
            #version 330 core
            void main() {
              gl_FragColor = vec4(1.0, 1.0, 1.0, 0.6);
            }
            '''
        )
        self.geom = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, '3f', 'in_vertex')],
            mode = moderngl.LINES
        )
        _, _, sw, sh = self.ctx.viewport
        self.sw = sw
        self.sh = sh
        self.update_size(hsize, vsize)
        
    def update_size(self, hsize, vsize):
        self.prog['hscale'] = hsize / self.sw
        self.prog['vscale'] = vsize / self.sh
        self.prog['dy'] = -1 + 2.0 * vsize / self.sh

    def render(self):
        self.geom.render()


        
        
class Scene:

    def __init__(self, filepath, target, display_surface):

        self.display_surface = display_surface
        
        hc = HilbertCurve(p = 16, n = 2) # 2^np points so with n = 2 we need p = 16

        # Load IP addresses
        cachepath = filepath + ".cache"
        if os.path.exists(cachepath):
            print("Loading cache...")
            addrs = pd.read_pickle(cachepath)
            locs = np.stack((addrs.x, addrs.y), axis = 1)
        else:
            print(f"Loading from {filepath}...")
            addrs = pd.read_csv(filepath)
            addrs['i'] = addrs.addr.apply(lambda x: int(ipaddress.ip_address(x)))
            
            print("Loaded addresses.")
            
            # x, y in [0, 2^16]
            locs = np.array(hc.points_from_distances(addrs.i)).astype('f4')
            
            # Normalize to [-1, 1]
            locs /= 2 ** (16 - 1)
            locs -= 1
            addrs['x'] = locs[:,0]
            addrs['y'] = locs[:,1]
            print("Computed distances.")
            
            addrs.to_pickle(cachepath)

        # Setup graphics

        minAlpha = addrs.alpha.min()
        maxAlpha = addrs.alpha.max()

        vertData = np.concatenate((
            locs,
            np.stack([1.0 - COLOR_SCALE * (addrs.alpha - minAlpha) / (maxAlpha - minAlpha), addrs.r2], axis = 1)
        ), axis = 1).astype('f4')
        
        self.ctx = moderngl.get_context()

        self.program = self.ctx.program(
            vertex_shader='''
                #version 330 core

                uniform float zoom = 1.0;
                uniform float aspect = 1.0;
                uniform vec2 target = vec2(0,0);
            
                layout (location = 0) in vec4 in_vertex;

                varying vec2 color;

                void main() {
                    gl_Position = vec4((in_vertex.xy - target) * zoom, 0.0, 1.0);
                    gl_Position.x /= aspect;
                    color = in_vertex.zw;
                }
            ''',
            fragment_shader='''
#version 330 core

uniform float zoom = 1.0;
                
varying vec2 color;

vec3 color_map(float t) {

    const vec3 c0 = vec3(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
    const vec3 c1 = vec3(0.1050930431085774, 1.404613529898575, 1.384590162594685);
    const vec3 c2 = vec3(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
    const vec3 c3 = vec3(-4.634230498983486, -5.799100973351585, -19.33244095627987);
    const vec3 c4 = vec3(6.228269936347081, 14.17993336680509, 56.69055260068105);
    const vec3 c5 = vec3(4.776384997670288, -13.74514537774601, -65.35303263337234);
    const vec3 c6 = vec3(-5.435455855934631, 4.645852612178535, 26.3124352495832);

    return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
}
//  Function from Iñigo Quiles
//  https://www.shadertoy.com/view/MsS3Wc
vec3 hsb2rgb( in vec3 c ){
    vec3 rgb = clamp(abs(mod(c.x*6.0+vec3(0.0,4.0,2.0),
                             6.0)-3.0)-1.0,
                     0.0,
                     1.0 );
    rgb = rgb*rgb*(3.0-2.0*rgb);
    return c.z * mix(vec3(1.0), rgb, c.y);
}
            
void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    float alpha = 1.0; // zoom * 0.0001 / pow(length(coord), 4);
    float hue = color.x;
    vec4 c = vec4(0.0, 0.0, 0.0, alpha);
    // c.rgb = oklch2lsrgb(vec3(0.72, 0.15, hue));
    // c.rgb = color_map(hue);
    c.rgb = hsb2rgb(vec3(hue, 1.0, 1.0));
    gl_FragColor = c;
}
            
            ''',
        )

        self.vbo = self.ctx.buffer(vertData.tobytes())
        self.vao = self.ctx.vertex_array(
            self.program, [(self.vbo, '4f', 'in_vertex')],
            mode = self.ctx.POINTS
        )

        self.start_time = None
        self.zoom = 1
        self.target = np.array(hc.point_from_distance(int(ipaddress.ip_address(target)))).astype('f4')
        self.target /= 2 ** (16 - 1)
        self.target -= 1

        self.duration = 100 * 1000

        _, _, sw, sh = self.ctx.viewport
        self.aspect = sw / sh
        self.height_pix = sh
        self.program['aspect'] = self.aspect

        self.scale_text = TextPlane("", font_size = 52)

        self.legend_key = LegendKey(300, 50)
        self.l = 4

        self.color_scale = ColorScale()
        self.color_scale_min = TextPlane(f"{minAlpha :.2f}", font_size = 52, y = -0.9, x = 0.15)
        self.color_scale_max = TextPlane(f"{maxAlpha :.2f}", font_size = 52, y = -0.9, x = -0.15)
        self.color_scale_label = TextPlane(f"alpha", font_size = 52, y = -0.9, x = 0.225)
        

    def render(self):
        if self.start_time is None:
            self.start_time = pygame.time.get_ticks()
            
        self.ctx.clear()
        self.ctx.enable(moderngl.BLEND)
        

        if self.l < 32:
            sz = self.zoom * float(self.height_pix) / (2.0 ** (16.0))
            
            self.ctx.point_size = sz if sz > 1.0 else 1
            self.program['zoom'] = self.zoom
            self.program['target'] = self.target
            
            side_len = sz * 2 ** ((32 - self.l) / 2)
            if side_len > 800:
                self.l += 4
                side_len = sz * 2 ** ((32 - self.l) / 2)
                
            
            # pixel_pfx_len = 32 - np.log2(1.0 / sz) # what length of prefix does each pixel represent
            self.scale_text.update_text(f"/{self.l}")
            self.legend_key.update_size(side_len * 2, 50) # * 2 because the points are actually drawn 2x larger from -1 to 1...
        self.vao.render()
        self.scale_text.render()
        self.legend_key.render()
        self.color_scale.render()
        self.color_scale_min.render()
        self.color_scale_max.render()
        self.color_scale_label.render()

        cur_time = (pygame.time.get_ticks() - self.start_time) / self.duration
        self.zoom = 2 ** (28 * cur_time - 4)
        
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <file with list of source IPv4 addresses> <target address to zoom in on>")
        sys.exit()

    os.environ['SDL_WINDOWS_DPI_AWARENESS'] = 'permonitorv2'

    pygame.init()
    display_surface = pygame.display.set_mode((1920, 1080), flags=pygame.OPENGL | pygame.DOUBLEBUF | pygame.FULLSCREEN, vsync=True)
        
    scene = Scene(sys.argv[1], sys.argv[2], display_surface)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
    
        scene.render()
    
        pygame.display.flip()
