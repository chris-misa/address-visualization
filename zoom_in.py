"""
Script to render zoom-in on Hilbert-space animation
"""

import math
import os
import sys

import glm
import moderngl
import numpy as np
import pygame
from hilbertcurve.hilbertcurve import HilbertCurve
import ipaddress

class Scene:

    def __init__(self, filepath, target):

        # Load IP addresses

        addrs = np.array([int(ipaddress.ip_address(x)) for x in  np.loadtxt(filepath, dtype = str)])

        print("Loaded addresses...")
        
        hc = HilbertCurve(p = 16, n = 2) # 2^np points so with n = 2 we need p = 16

        locs = np.array(hc.points_from_distances(addrs)).astype('f4')

        # Normalize to [-1, 1]
        locs /= 2 ** (16 - 1)
        locs -= 1
        locs /= 2

        print("Computed distances...")

        # Setup graphics
        
        self.ctx = moderngl.get_context()

        self.program = self.ctx.program(
            vertex_shader='''
                #version 330 core

                uniform float zoom = 1.0;
                uniform vec2 target = vec2(0,0);
            
                layout (location = 0) in vec2 in_vertex;

                void main() {
                    gl_Position = vec4((in_vertex - target) * zoom, 0.0, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330 core

                void main() {
                    gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
                }
            ''',
        )

        self.vbo = self.ctx.buffer(locs.tobytes())
        self.vao = self.ctx.vertex_array(
            self.program, [(self.vbo, '2f', 'in_vertex')],
            mode = self.ctx.POINTS
        )

        self.zoom = 1.0
        self.target = np.array(hc.point_from_distance(int(ipaddress.ip_address(target)))).astype('f4')
        self.target /= 2 ** (16 - 1)
        self.target -= 1
        self.target /= 2


    def render(self):
        self.ctx.clear()
        self.ctx.point_size = 1
        self.program['zoom'] = self.zoom
        self.program['target'] = self.target
        self.zoom *= 1.005
        self.vao.render()
        
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <file with list of source IPv4 addresses> <target address to zoom in on>")
        sys.exit()

    os.environ['SDL_WINDOWS_DPI_AWARENESS'] = 'permonitorv2'

    pygame.init()
    pygame.display.set_mode((800, 800), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)
        
    scene = Scene(sys.argv[1], target = sys.argv[2])

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
    
        scene.render()
    
        pygame.display.flip()
