var canvas = document.querySelector("#myChart");
var ctx = canvas.getContext('2d');
var loader = document.querySelector("#loader");
var emotions_ready = true;
var summary_ready = true;
var loader_started = false;

function update_loader(){
    if(emotions_ready && summary_ready){
        loader.style.display = "none";
        loader_started = false;
    }else if(!loader_started){
        loader.style.display = "inline-block"
        loader_started = true;
    }
}


eel.expose(on_emotions_ready)
function on_emotions_ready(data_str, emotions_str){
    console.log("received emotion scores")
    emotions_ready = true;
    var data = data_str.split(" ").map(Number);
    var labels = emotions_str.split(" ");
    console.log(data);
    console.log(labels);
//    data = [66000,	41000,	40000,	33000,	28000,	13000,	17000, 30000];
//    labels = ["Java",	"Python",	"JavaScript",	"C++",	"C#",	"PHP",	"Perl", "new"];
     ctx.clearRect(0, 0, canvas.width, canvas.height);
    // End Defining data
    var myChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Emotion scores', // Name the series
                data: data, // Specify the data values array
                backgroundColor: [ // Specify custom colors
                    'rgba(255, 99, 132, 0.2)',
                    'rgba(54, 162, 235, 0.2)',
                    'rgba(255, 206, 86, 0.2)',
                    'rgba(75, 192, 192, 0.2)',
                    'rgba(153, 102, 255, 0.2)',
                    'rgba(255, 159, 64, 0.2)',
                    'rgba(255, 172, 99, 0.2)',
                    'rgba(255, 159, 255, 0.2)'
                ],
                borderColor: [ // Add custom color borders
                    'rgba(255,99,132,1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 206, 86, 1)',
                    'rgba(75, 192, 192, 1)',
                    'rgba(153, 102, 255, 1)',
                    'rgba(255, 159, 64, 1)',
                    'rgba(255, 159, 99, 1)'
                ],
                borderWidth: 1 // Specify bar border width
            }]
        },
        options: {
          responsive: true, // Instruct chart js to respond nicely.
          maintainAspectRatio: true, // Add to prevent default behaviour of full-width/height
        }
    });
    update_loader();
}

on_emotions_ready("","")

eel.expose(on_summary_ready);
function on_summary_ready(summary_embedding, summary_tokens){
    summary_ready = true;
    console.log("received summary");
    document.querySelector("#summary_embeddings").innerHTML = summary_embedding;
    document.querySelector("#summary_tokens").innerHTML = summary_tokens;
    update_loader();
}

function create_summary(){
    console.log("Preparing Summary request...")
    summary_ready = false;
    emotions_ready = false;
    text = document.querySelector("#text_input").value;
    text = text.replace(/(\r\n|\n|\r)/gm, "");
    text = text.trim();

    compression = parseFloat(document.querySelector("#compression").value) / 100;
    if(compression < 0){
        compression = -compression;
    }

    if(compression > 100){
        compression = 100;
    }

    console.log("Compression" + compression)

    if(text==""){
        document.querySelector("#error_msg").style.display = "inline-block";
        return;
    }else{
        document.querySelector("#error_msg").style.display = "none";
    }
    console.log("Sending Summary request...")
    eel.summarize(text, compression);
    eel.detect_emotions(text)
    update_loader();
}