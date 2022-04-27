1. Requiere de Python instalado
2. pip install flask en un python, en git bash no me funciono. En Mac o Linux se usa pip3.
    
    pip install pymongo pandas "pymongo[srv]”

3. Para la ejecución usamos: 
   a) `py -3 -m venv myenv` si no existe un venv: virtual environment 

   b) .\venv\Scripts\activate.bat para correr el entorno

   c) export FLASK_APP=<nombre_del_archivo> (app.py en este caso) 
    
      set FLASK_APP=app.py (en Windows)
    
   d) flask run
