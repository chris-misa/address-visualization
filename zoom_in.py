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

class Scene:

    def __init__(self, filepath, target):
        
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
            np.stack((260.0 - 200.0 * ((addrs.alpha - minAlpha) / (maxAlpha - minAlpha)) ** 0.5, addrs.r2), axis = 1)
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

// From: https://gist.github.com/akella/059d9877b90f966c9181ffa2bc5ffd65

float fixedpow(float a, float x)
{
    return pow(abs(a), x) * sign(a);
}

float cbrt(float a)
{
    return fixedpow(a, 0.3333333333);
}

vec3 lsrgb2oklab(vec3 c)
{
    float l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
    float m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
    float s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;

    float l_ = cbrt(l);
    float m_ = cbrt(m);
    float s_ = cbrt(s);

    return vec3(
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    );
}

vec3 oklab2lsrgb(vec3 c)
{
    float l_ = c.r + 0.3963377774 * c.g + 0.2158037573 * c.b;
    float m_ = c.r - 0.1055613458 * c.g - 0.0638541728 * c.b;
    float s_ = c.r - 0.0894841775 * c.g - 1.2914855480 * c.b;

    float l = l_ * l_ * l_;
    float m = m_ * m_ * m_;
    float s = s_ * s_ * s_;

    return vec3(
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    );
}

vec3 oklch2lsrgb(vec3 c)
{
    float h = radians(c.b);
    return oklab2lsrgb(vec3(
        c.r,
        c.g * cos(h),
        c.g * sin(h)
    ));
}
            
void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    float alpha = 1.0; // zoom * 0.0001 / pow(length(coord), 4);
    float hue = color.x;
    vec4 c = vec4(0.0, 0.0, 0.0, alpha);
    c.rgb = oklch2lsrgb(vec3(0.72, 0.15, hue));
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

        self.duration = 120 * 1000

        _, _, sw, sh = self.ctx.viewport
        self.aspect = sw / sh
        self.program['aspect'] = self.aspect
        

    def render(self):
        if self.start_time is None:
            self.start_time = pygame.time.get_ticks()
            
        self.ctx.clear()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.point_size = 1
        self.program['zoom'] = self.zoom
        self.program['target'] = self.target
        self.vao.render()

        cur_time = (pygame.time.get_ticks() - self.start_time) / self.duration
        self.zoom = 2 ** (28 * cur_time - 4)
        
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <file with list of source IPv4 addresses> <target address to zoom in on>")
        sys.exit()

    os.environ['SDL_WINDOWS_DPI_AWARENESS'] = 'permonitorv2'

    pygame.init()
    pygame.display.set_mode((1920, 1080), flags=pygame.OPENGL | pygame.DOUBLEBUF | pygame.FULLSCREEN, vsync=True)
        
    scene = Scene(sys.argv[1], sys.argv[2])

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
    
        scene.render()
    
        pygame.display.flip()
