function page_load() {
	$('#reason-div').hide();
}

function ask_reason() {
	$('#reason-div').show();
}

function submit_msg() {
	//document.body.style.background = "red";
	msg = $('#msg-field').val();
	$('#msg-field').val('');
	$('#chatboard').append("<div class='user'>"+msg+"</div>");
	$.get( "/msg", {msg:msg}, function( resp ) {
		sents = resp.split('|')
		console.log(sents)
		for (i=0; i<sents.length; i++) {
			$('#chatboard').append("<div class='bot'>"+sents[i]+"</div>");
		}
	});
}

$('input').keyup(function(e){
    if(e.keyCode == 13)
    {
        submit_msg();
    }
});