from flask import Flask, send_from_directory

import pandas as pd
import numpy as np
import os
from io import BytesIO
import base64
import cv2

import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_table
from dash.dependencies import Input, Output, State

import plotly
from plotly.subplots import make_subplots
import plotly_express as px

from io import BytesIO
import base64
import matplotlib.pyplot as plt
import json

static_image_route = '/static/'
filespath='/project/DSone/jaj4zcf/Videos/'


app = Flask(__name__)


application = dash.Dash(__name__, server=app,url_base_pathname='/')


videos=[]

for file in os.listdir('/project/DSone/jaj4zcf/Videos/ResultsSodiqCSV'):
    if file.endswith(".csv"):
        videos.append(file[0:-4])
        
videos



def buildimages(input_value, n_val, vid, table):    
    frames=[i['index'] for i in table]  #[item for item in range(-10,10)] #[-3,-2,-1,0,1,2,3]
    
    row=n_val
    images=[]
    labels=[]
    
    for i,offset  in enumerate(frames):
        impath='/project/DSone/jaj4zcf/Videos/v'+str(vid)[-2:]+'/'+str(offset)+'.png'    ## may need to be updated for final!
        impathdirect='/vids/v'+str(vid)[-2:]+'/'+str(offset)+'.png'
        ## Only add if file exists
        try:
            #encoded_image = base64.b64encode(open(impath, 'rb').read()).decode("ascii").replace("\n", "")
            #images.append(html.Td(html.Img(src='data:image/png;base64,{}'.format(encoded_image), style={'width': '100%'})))
            images.append(html.Td(html.Img(src=impathdirect, style={'width': '100%'})))
            ## add labels
            if offset==0:
                labels.append(html.Td('Selected Frame: ' + str(row)))
            else:
                labels.append(html.Td('Frame: ' + str(offset)))
        except:
            'poo'
   
    images=html.Table([html.Tr(labels),html.Tr(images)])
    
    return images




vidLabes=[]
for vid in videos:
    vidLabes.append({'label': 'Model Result: '+ str(vid), 'value':str(vid)})

labelsdf=pd.read_csv('/project/DSone/jaj4zcf/Videos/ResultsSodiqCSV/'+str(videos[2])+'.csv')
labelsdf=labelsdf.replace(np.nan, '', regex=True)

labelsdf=labelsdf.reset_index()


timeline=dcc.Graph(
        id='timeline',
        figure= px.scatter(data_frame=labelsdf, y='Pathology', x='index', color='TractSect1', hover_name="time" )
    )

graph_height=300

figure= px.scatter(labelsdf, y='Pathology', x='index', color='TractSect1', hover_name="time", height=graph_height )
figure.layout={
                'clickmode': 'event+select'
            }

videoSelect=dcc.Dropdown(
        id='videoSelect',
        options=vidLabes,
        value=videos[0]
    )
PAGE_SIZE=10





styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}


application.layout = html.Div(
    [
        html.H2('Deep VCE Results Explorer'),

        html.H5('Choose a Video and Model Prediction Result to Begin:'),
        videoSelect,  
        html.H5('Select points or peaks to explore results:'),
    dcc.Loading(
                    id="loading-2",
                    children=[html.Div([timeline])],
                    type="circle",
                ),
    
    html.H5('Video Frames:'),

    
    html.Div(id='imagecontainer', style={'display':'inline'}),

    html.Div([
    html.Button('Previous', id='prev', n_clicks=0),
    html.Button('Next', id='next', n_clicks=0),
    html.P(id='test'),
    ], style={'display':'inline'}), 
    
    html.H5('Use the Data Table Below to explore Results - rows are clickable.'),
    dash_table.DataTable(
    id='table',
    editable=True,
    page_size = PAGE_SIZE,
    sort_action = 'native',
    filter_action = 'native',
    row_selectable='single',
    
    columns=[{"name": i, "id": i} for i in labelsdf[['index','sectNorm', 'time', 'TractSect1', 'Pathology', 'Notes', 'small bowelabNormal']].columns],
    #columns[-1][-1]={'name': 'small bowelabNormal', 'id': 'Prob. SB abNorm.'},
    dropdown={
            'TractSect1': {
                'options': [
                    {'label': i, 'value': i}
                    for i in labelsdf['TractSect1'].unique()  #['colon', 'small bowell', 'stomach', 'pylorus']
                ]
            }},    
    data=labelsdf[['index','sectNorm', 'time', 'TractSect1', 'Pathology', 'Notes', 'small bowelabNormal']].to_dict('records')
    ),
    html.Div(id='table-var', style={'display': 'none'}),  #where to store the table values.
    html.Div(id='offset-var', children=[0], style={'display': 'none'}),  #where to store the offset from table to index. 

    html.Div(id='frame-var',children=[0], style={'display': 'none'}),  #store the current frame 
    html.Div(id='prev-click',children=[0], style={'display': 'none'}),
    
    html.Div(id='num-prev',children=[0], style={'display': 'none'}),
    html.Div(id='num-next',children=[0], style={'display': 'none'})
                      ])


@application.callback(
     Output('test', 'children'),
    [Input('table', 'selected_rows'), Input('videoSelect', 'value')])
def update_image_table(selected_rows, value):
    if value[-2].isnumeric():
        video=value[-2:]
    else:
        video=value[-1]
    if selected_rows is None:
        return 'no selection'
    else:
        return selected_rows



@application.callback(
    Output('imagecontainer', 'children'),
    [Input('table', 'selected_rows'), Input('videoSelect', 'value'), Input('offset-var', 'children'), Input('table', 'derived_viewport_data') ])
def update_image_div(selected_rows, value, offset, table):
    if value[-2].isnumeric():
        video=value[-2:]
    else:
        video=value[-1]
    
    if selected_rows is None:
        impath='None'
        res='nothing'
        return None
    else:
        row=selected_rows[0]+offset
        images = buildimages('poo', row, video, table)

        return images

    return 





####
### Front and Back Buttons

@application.callback(
    [Output('frame-var', 'children'), Output('num-next', 'children')],
    [Input('timeline', 'clickData'), Input('next', 'n_clicks')],
    [State('frame-var', 'children'),State('num-next', 'children') ])
def update_output(clickData, n_next_clicks, frame,n_next_state):
    
    # Check to see if num of clicks changed
    if n_next_clicks != n_next_state:
        return [int(frame[0]) + 1], n_next_clicks
    else: 
        return [clickData['points'][0]['x']], n_next_clicks   
    #except:
       #return 0,0    


## Change table selection based on current frame and using offset (table does not start at 0)
@application.callback(
    Output('table', 'selected_rows'),
    [Input('frame-var', 'children'), Input('offset-var', 'children')])
def display_click_data(frame, offsetvar):
    try:    
        val=0
        val=frame[0]-offsetvar
        return [val] #[int(test['points'][0]['x'])]
    except:
        return [0]
    
## Set current page of table so that selected frame is visisble. 
@application.callback(
    Output('table', 'page_current'),
    [Input('table', 'derived_virtual_selected_rows')])   # Input('table', "derived_virtual_selected_rows")
def display_click_data(sel_rows):
    
    try:
        if sel_rows[0]>3:
            return sel_rows[0]/PAGE_SIZE-3/PAGE_SIZE
        else:
            return sel_rows[0]/PAGE_SIZE
    except:
        return None
    

# Load new video. 
@application.callback([Output('timeline', 'figure'), Output('table', 'data') , Output('offset-var', 'children'),Output('table-var', 'children')], 
                      [Input('videoSelect', 'value')])
def return_data(value):

    vid=value
    try:
        labelsdf=pd.read_csv('/project/DSone/jaj4zcf/Videos/ResultsSodiqCSV/'+str(vid)+'.csv')
    except:
        labelsdf=pd.read_csv('/project/DSone/jaj4zcf/Videos/ResultsSodiqCSV/'+videos[0]+'.csv')
    
    labelsdf=labelsdf.replace(np.nan, '', regex=True)

    offsetvar=labelsdf['index'].min() 
    
    ## Update Timeline
    try:
        labelsdfBar=labelsdf[labelsdf['small bowelabNormal']>=.3]

        labelsdfScat=labelsdf[labelsdf['sectNorm']=='small bowelabNormal']

        colorsIdx = {'mouth': 'rgb(240,128,128)', 'stomach': 'rgb(255,160,122)', 'pylorus': 'rgb(100,149,237)', 'small bowel': 'rgb(147,112,219)', 'colon': 'rgb(205,133,63)'}
        cols      = labelsdf['TractSect1'].map(colorsIdx)

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_bar(secondary_y=False, y=100*labelsdfBar['small bowelabNormal'], width=1, x=labelsdfBar['index'],text=labelsdfBar['small bowelabNormal'], marker_color='red', opacity=1)
        fig.add_scatter(secondary_y=True, mode='markers',y=labelsdfScat.Pathology, x=labelsdfScat['index'], text=labelsdfScat.time, customdata=labelsdfScat['small bowelabNormal'], marker=dict(size=3, color=cols), 
                       hovertemplate="Pathology: %{y}<br>index: %{x}<br>time: %{text}<br>Prob. Abnormal: %{customdata}<extra></extra>   ")

        fig.update_layout(plot_bgcolor='rgb(250,250,250)', yaxis_title="Probability of Abnormality (%)", margin={'t': 5, 'b':5}) #fig
    
    except:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
   
    labelsdf=labelsdf[['index','sectNorm', 'time', 'TractSect1', 'Pathology', 'Notes', 'small bowelabNormal']]
    return fig, labelsdf.to_dict('records') , offsetvar, labelsdf.to_json(orient='split')      



@app.route('/vids/<path:path>')
def send_jss(path):
    pos=-(path[::-1].find('/'))
    filename=path[pos::]
    path=path[0:pos]
    print('poop')
    return send_from_directory(filespath+path, filename)

@app.route('/v1/<path:path>')
def send_js(path):
    return send_from_directory(filespath+'v1/', path)


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
