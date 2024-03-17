const express = require('express')
const cors = require('cors')
const app = express()
const path = require('path');
const {
  spawn
} = require('child_process');


app.set('view engine', 'ejs');
app.use(express.static(path.join(__dirname, '/public')));



app.use(cors())


app.get("/", (req, res) => {
  res.render('index', {
    title: 'Home Page'
  });
})

app.get('/molecule', (req, res) => {
  const complexity = req.query.complexity;
  console.log('Molecule complexity:', complexity);



  var dataToSend;
  // spawn new child process to call the python script
  const python = spawn('python3', ['script.py', complexity]);
  // collect data from script
  python.stdout.on('data', function (data) {
    console.log('Pipe data from python script ...');
    dataToSend = data.toString();
  });


  python.on('close', (code, err) => {
    console.log(`child process close all stdio with code ${code}`);
    console.log(dataToSend)
    // send data to browser
    res.render('molecule.ejs', {
      smiles: dataToSend
    })
  });

  if (python.stderr !== null) {
    python.stderr.on('data', (data) => {
      console.log(data.toString());
    });
  }

});

const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`CORS-enabled web server listening on port ${port}`)
})