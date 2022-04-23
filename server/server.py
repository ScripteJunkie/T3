from flask import Flask, render_template, Response
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0 # Default cache expiration is 12 hours

@app.route('/')
def main():
   return render_template('main.html')

if __name__ == '__main__':
   app.run(host='localhost', port=8000)
   app.run(debug=True)
#
# #Use to create local host
# import http.server
# import socketserver
#
# PORT = 8000
#
# Handler = http.server.SimpleHTTPRequestHandler
# Handler.extensions_map.update({
#       ".js": "application/javascript",
# });
#
# httpd = socketserver.TCPServer(("", PORT), Handler)
# httpd.serve_forever()