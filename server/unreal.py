import unreal

level_path = '/Code/Unreal/Senz'
level = unreal.find_asset(None, name=level_path)

static_meshes = unreal.GameplayStatics.get_all_actors_of_class(level, unreal.StaticMeshActor)
print(static_meshes)