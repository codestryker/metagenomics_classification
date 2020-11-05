from flask import Flask,flash,request,render_template,redirect,url_for
from app.torch_utils import *


app= Flask(__name__)
app.secret_key = 'sam'

@app.route('/')
def index():
    return render_template('index.html',error=False)

@app.route('/predict',methods=['POST'])
def predict():
    seq=request.form['code']
    load()
    try:
        sys.stdout.flush()
        result = get_predict(seq)
        
    except Exception as e:
        print("error:",e)
        return render_template('index.html',error=True)
    headings=('Kingdom','Phylum','Class','Order','Family','Genus','Species')
    if len(seq)>50:
      return render_template('result.html',headings=headings,data=result,error=False)
    else:
      flash('Your input length should be greater than 50')
      return redirect(url_for('index'))
    

if __name__=='__main__':
    app.run()
    
    
    
    
