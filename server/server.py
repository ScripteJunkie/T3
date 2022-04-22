from flask import Flask, render_template, Response
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0 # Default cache expiration is 12 hours

@app.route('/')
def main():
   return render_template('main.html')
    
if __name__ == '__main__':
   app.run(host='0.0.0.0', port=8090)
   # app.run(debug=True)