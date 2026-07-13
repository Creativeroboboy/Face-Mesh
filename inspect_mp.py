import importlib
import mediapipe
print('mediapipe', mediapipe.__file__)
mp = importlib.import_module('mediapipe.tasks.python')
print('tasks.python', getattr(mp, '__file__', getattr(mp, '__path__', None)))
print('vision attr', hasattr(mp, 'vision'))
if hasattr(mp, 'vision'):
    v = mp.vision
    print('vision module', getattr(v, '__file__', getattr(v, '__path__', None)))
    print('vision members', [name for name in dir(v) if name.startswith('Face') or name.startswith('drawing') or name.endswith('RunningMode')])
    import mediapipe.tasks.python.vision.face_landmarker as fl
    print('face_landmarker file', fl.__file__)
    print('FaceLandmarker methods', [name for name in dir(fl.FaceLandmarker) if name.startswith('create') or name.startswith('detect') or name.startswith('get')])
