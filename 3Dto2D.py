import bpy
import os
import mathutils
import random
import math

path = "Z:/Chair,Desk 62OBJ"
folder_lst = os.listdir(path)

def calculate_bounding_box(obj):
    bbox_center = sum((mathutils.Vector(vertex) for vertex in obj.bound_box), mathutils.Vector()) / 8
    bbox_dimensions = max(obj.dimensions)
    camera_location = mathutils.Vector((0, 0, bbox_dimensions * 4))
    return camera_location


def set_origin_to_bounding_box_center(obj):
    bbox_center_local = sum((mathutils.Vector(vertex) for vertex in obj.bound_box), mathutils.Vector()) / 8
    bbox_center_world = obj.matrix_world @ bbox_center_local
    bpy.context.scene.cursor.location = bbox_center_world
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
        

for i in range(len(folder_lst)):
    second_F = []
    path2 = path + "/" + folder_lst[i]
    second_F = os.listdir(path2)
    for j in range(len(second_F)):
        file_obj = []
        path3 = path2 + "/" + second_F[j]
        fileList = os.listdir(path3)
        file_obj = [file for file in fileList if file.endswith(".obj")]
        for file in file_obj:
            bpy.ops.import_scene.obj(filepath = path3 + "/" + file)
            for obj in bpy.context.scene.objects:
                if obj.type != 'CAMERA' and obj.type != 'LIGHT':
                    obj.select_set(True)
            
            # 선택한 오브젝트들을 활성화 객체로 설정
            if bpy.context.selected_objects:
                bpy.context.view_layer.objects.active = bpy.context.selected_objects[-1]
            bpy.ops.object.join()
            
            # Create camera
            camera_data = bpy.data.cameras.new("Camera")
            camera = bpy.data.objects.new("Camera", camera_data)
            bpy.context.scene.collection.objects.link(camera)
            bpy.context.scene.camera = camera
            
            # create light
            light_data = bpy.data.lights.new("Light", type='SUN')
            light = bpy.data.objects.new("Light", light_data)
            bpy.context.scene.collection.objects.link(light)
            light.location = (0, 0, 10)
            light.data.energy = 5.0

            bpy.context.view_layer.objects.active = camera
            
            for i in range(8):
                for obj in bpy.context.scene.objects:
                    if obj.type != 'CAMERA' and obj.type != 'LIGHT':
                        
                        set_origin_to_bounding_box_center(obj)
                        obj.location = (0, 0, 0)
                        
                        rx = random.uniform(0, 360)
                        ry = random.uniform(0, 360)
                        rz = random.uniform(0, 360)
                        obj.rotation_euler = (math.radians(rx), math.radians(ry), math.radians(rz))

                        obj.location = (0, 0, 0)
                                                
                        camera_location = calculate_bounding_box(obj)
                
                        camera.location = camera_location
                        
                        bpy.context.scene.camera = camera
                        bpy.context.scene.render.filepath = str(path3) + f'/Camera{i}.png'
                        bpy.ops.render.render(write_still=True)
                    
                        
                bpy.ops.object.select_all(action='DESELECT')

            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete(use_global=False)