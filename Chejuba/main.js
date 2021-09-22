function req1() {
  fetch('https://jsonplaceholder.typicode.com/posts')
    .then(response => response.json())
    .then(json => {
        let myArray=[];
        let jsonData=json;
        //console.log(jsonData);
        for(var i=0;i<jsonData.length;i++){
            let subdata=jsonData[i];
            let subdataId=subdata.id;
            if(subdataId % 3==0 && subdataId % 5==0){
                subdata.category='magic';
            }
            else if(subdataId % 5==0){
                subdata.category='fifths';
            }
            else if(subdataId % 3==0){
                subdata.category='thirds';
            }
            myArray.push(subdata[i]);
        }
        console.log(jsonData);
    //   const title = json.title;
    //   const body = json.body;
        document.getElementById("printTitle").innerHTML = jsonData;
    //   document.getElementById("printBody").innerHTML = body;
    });
}

