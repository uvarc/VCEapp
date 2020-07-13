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

import dash_table.FormatTemplate as FormatTemplate
from dash_table.Format import Sign

live=True


app = Flask(__name__)


application = dash.Dash(__name__, server=app,url_base_pathname='/')
videos=[]

for file in os.listdir('/project/DSone/jaj4zcf/Videos/ResultsSodiqCSV'):
    if file.endswith(".csv"):
        videos.append(file[0:-4])
        
videos


def buildTopImage(center_frame, n_images, vid):    
    if n_images[0]=='1':
        frames=[0]
    if n_images=='3':
        frames=[-1,0,1]
    if n_images=='5':
        frames=[-2,-1,0,1,2]
    if n_images=='7':
        frames=[-3,-2,-1,0,1,2,3]
        
    images=[]
    labels=[]
    
    for i,offset  in enumerate(frames):
        impath='/project/DSone/jaj4zcf/Videos/v'+str(vid)[-2:]+'/'+str(int(center_frame) + offset)+'.png'    ## may need to be updated for final!
        impathdirect='/vids/v'+str(vid)[-2:]+'/'+str(center_frame + offset)+'.png'
        ## Only add if file exists
        try:
            if live==True:
                images.append(html.Td(html.Div(html.Img(src=impathdirect, style={'max-width': '250px','width': '100%'}), className='zoom')))
            else:
                encoded_image = base64.b64encode(open(impath, 'rb').read()).decode("ascii").replace("\n", "")
                images.append(html.Td(html.Div(html.Img(src='data:image/png;base64,{}'.format(encoded_image), style={'max-width': '250px','width': '100%'}), className='zoom')))
            
            ## add labels
            if offset==0:
                labels.append(html.Td('Selected Frame: ' + str(center_frame)))
            else:
                labels.append(html.Td('Frame: ' + str(center_frame + offset)))
        except:
            'poo'
   
    image_table=html.Table(html.Tr(images),style={'text-align':'center','margin':'auto'} )
    
    return image_table



def buildimages(vid, table):    
    frames=[i['index'] for i in table]  #[item for item in range(-10,10)] #[-3,-2,-1,0,1,2,3]
    
    images=[]
    labels=[]
    
    for i,offset  in enumerate(frames):
        impath='/project/DSone/jaj4zcf/Videos/v'+str(vid)[-2:]+'/'+str(offset)+'.png'    ## may need to be updated for final!
        impathdirect='/vids/v'+str(vid)[-2:]+'/'+str(offset)+'.png'
        ## Only add if file exists
        try:
            if live==True:
                images.append(html.Tr(html.Td(html.Div(html.Img(src=impathdirect, style={'width': '100%'}), className='zoom'))))
            else:
                encoded_image = base64.b64encode(open(impath, 'rb').read()).decode("ascii").replace("\n", "")
                images.append(html.Tr(html.Td(html.Div(html.Img(src='data:image/png;base64,{}'.format(encoded_image), style={'width': '100%'}), className='zoom'))))
            
            ## add labels
            if offset==0:
                labels.append(html.Td('Selected Frame: ' + str(row)))
            else:
                labels.append(html.Td('Frame: ' + str(offset)))
        except:
            'poo'
   
    images=html.Table(images,style={'width': '99%', 'float':'left'} )
    
    return images




vidLabes=[]
for vid in videos:
    vidLabes.append({'label': 'Model Result: '+ str(vid), 'value':str(vid)})

labelsdf=pd.read_csv('/project/DSone/jaj4zcf/Videos/ResultsSodiqCSV/'+str(videos[2])+'.csv')
labelsdf=labelsdf.replace(np.nan, '', regex=True)

labelsdf=labelsdf.reset_index()


timeline=dcc.Graph(
        id='timeline' )

graph_height=300



videoSelect=dcc.Dropdown(
        id='videoSelect',
        options=vidLabes,
        value=videos[0]
    )
PAGE_SIZE=10


COLUMNS=[{"name": i, "id": i} for i in labelsdf[['index','sectNorm', 'time', 'TractSect1', 'Pathology', 'Notes', 'small bowelabNormal']].columns]

COLUMNS[6].update({ "name":"Prob Abnormal",'type': 'numeric', 'format': FormatTemplate.percentage(1)})

COLUMNS[3].update({ "presentation": "dropdown"})

table=html.Div(dash_table.DataTable(
    id='table',
    editable=True,
    page_size = PAGE_SIZE,
    sort_action = 'native',
    filter_action = 'native',
    row_selectable='single',
    data=labelsdf[['index','sectNorm', 'time', 'TractSect1', 'Pathology', 'Notes', 'small bowelabNormal']].to_dict('records'),
    columns=COLUMNS,
    #columns[-1][-1]={'name': 'small bowelabNormal', 'id': 'Prob. SB abNorm.'},
    dropdown={
            'TractSect1': {
                'options': [
                    {'label': i, 'value': i}
                    for i in ['colon', 'small bowel', 'stomach', 'pylorus', 'esophagus']
                ]
            }},
    style_data_conditional=[
        {
            'if': {
                'filter_query': '{{small bowelabNormal}} > {}'.format(.5),
            },
            'backgroundColor': '#fff3f5',
        },
    ],
    
    style_cell={
        'whiteSpace': 'normal',
        'height': '40px',
        'overflow': 'hidden',
        'textOverflow': 'ellipsis',
        'maxWidth': 0,
        
    },
    ),style={'width':'auto','overflow':'hidden'})


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
    html.Div(id='scrub_display', style={'display':'inline-block', 'width':'99%', 'margin':'auto'}),
    dcc.Loading(
                    id="loading-2",
                    children=[html.Div([timeline], style={'max-height': '65px'})],
                    type="circle",
                ),
    dcc.Slider(
        id='scrub_frame',
        min=0,
        max=labelsdf.shape[1],
        step=1,
        value=0,
    ), 
    dcc.Dropdown(
        id='n_images',
        options=[
            {'label': '1', 'value': '1'},
            {'label': '3', 'value': '3'},
            {'label': '5', 'value': '5'},
            {'label': '7', 'value': '7'}
        ],
        value='1'
    ),
    

    html.Div([
    html.Button('Previous', id='prev', n_clicks=0),
    html.Button('Next', id='next', n_clicks=0),
    html.P(id='test'),
    ], style={'display':'inline'}), 
    
    html.H5('Use the Data Table Below to explore Results - rows are clickable.'),
    
    html.Div([
        html.Div( [
            html.Div(id='imagecontainertop', style={'height':'80px'}),
            html.Div(id='imagecontainer', style={'height':'1000px'}),
        ], id='imagecontainerwrap', style={'width':'100px','float':'left','display':'inline-block'}),
            table
        ],
        style={'display':'inline-block', 'width':'100%'}),
        
    dcc.Slider(
        id='imSize',
        min=100,
        max=512,
        step=1,
        value=100,
    ),  
    
    dcc.Slider(
        id='num_rows',
        min=3,
        max=20,
        step=1,
        value=10,
    ), 
        
    html.Div(id='table-var', style={'display': 'none'}),  #where to store the table values.
    html.Div(id='offset-var', children=[0], style={'display': 'none'}),  #where to store the offset from table to index. 

    html.Div(id='frame-var',children=[0], style={'display': 'none'}),  #store the current frame 
    html.Div(id='prev-click',children=[0], style={'display': 'none'}),
    
    html.Div(id='num-prev',children=[0], style={'display': 'none'}),
    html.Div(id='num-next',children=[0], style={'display': 'none'}),
        
    html.Div(id='abnormal_probs',children=[0], style={'display': 'none'}),
    html.P(id='abnormals',children=[0])   
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
    [ Input('videoSelect', 'value'), Input('table', 'derived_viewport_data') ])
def update_image_div(value, table):
    if value[-2].isnumeric():
        video=value[-2:]
    else:
        video=value[-1]
    
    try:
        images = buildimages(video, table)
        return images
    except:
        return None

@application.callback(
    Output('scrub_display', 'children'),
    [Input('scrub_frame', 'value'), Input('videoSelect', 'value'), Input('n_images', 'value')])
def update_image_div(center_frame, vid, n_images):
    if vid[-2].isnumeric():
        vid=vid[-2:]
    else:
        vid=vid[-1]
    return buildTopImage(center_frame, n_images, vid)

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
    Output('scrub_frame', 'value'),
    [Input('timeline', 'clickData')])
def display_click_data(clickData):
    try:    
        val=clickData['points'][0]['x']
        return val #[int(test['points'][0]['x'])]
    except:
        return 0

    
    
    
    
    
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
    

@application.callback(
    [Output('table', 'style_data'), Output('imagecontainerwrap', 'style')],
    [Input('imSize', 'value')])   # Input('table', "derived_virtual_selected_rows")
def tablepicsize(value):
    
    cellwidth=str(value)+'px'
    style_cell={'whiteSpace': 'normal', 'height': cellwidth} 
    style_wrap={'width':cellwidth,'float':'left','display':'inline-block'}
    
    return style_cell, style_wrap

@application.callback(
    Output('table', 'page_size'),
    [Input('num_rows', 'value')])   # Input('table', "derived_virtual_selected_rows")
def pages(value):
    return value

## Callback to get abnormal frames - by threshold. 

@application.callback(
    Output('abnormals', 'children'),
    [Input('abnormal_probs', 'children')])   # Input('table', "derived_virtual_selected_rows")
def return_abnormalframes(indexes):
    indexes=pd.read_json(indexes)
    test=indexes[indexes['small bowelabNormal']>.9]['index'].to_json()
    return test


### Callback must


@application.callback([Output('timeline', 'figure'), 
                       Output('table', 'data') , Output('offset-var', 'children'),Output('table-var', 'children'),
                       Output('scrub_frame', 'min') , Output('scrub_frame', 'max'), 
                       Output('abnormal_probs', 'children')], 
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
        #labelsdfBar=labelsdf[labelsdf['small bowelabNormal']>=.3]

        #labelsdfScat=labelsdf[labelsdf['sectNorm']=='small bowelabNormal']

        #colorsIdx = {'mouth': 'rgb(240,128,128)', 'stomach': 'rgb(255,160,122)', 'pylorus': 'rgb(100,149,237)', 'small bowel': 'rgb(147,112,219)', 'colon': 'rgb(205,133,63)'}
        #cols      = labelsdf['TractSect1'].map(colorsIdx)
        #fig = make_subplots(specs=[[{"secondary_y": True}]])
        #fig.add_bar(secondary_y=False, y=100*labelsdfBar['small bowelabNormal'], width=1, x=labelsdfBar['index'],text=labelsdfBar['small bowelabNormal'], marker_color='red', opacity=1)
        #fig.add_scatter(secondary_y=True, mode='markers',y=labelsdfScat.Pathology, x=labelsdfScat['index'], text=labelsdfScat.time, customdata=labelsdfScat['small bowelabNormal'], marker=dict(size=3, color=cols), 
        #               hovertemplate="Pathology: %{y}<br>index: %{x}<br>time: %{text}<br>Prob. Abnormal: %{customdata}<extra></extra>   ")
        #fig.update_layout(plot_bgcolor='rgb(250,250,250)', yaxis_title="Probability of Abnormality (%)", margin={'t': 5, 'b':5}) #fig
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=labelsdf['small bowelabNormal'],
            x=labelsdf['index'],
            y=np.ones(len(labelsdf['index'])),
            colorscale='reds', showscale=False) )
        fig.update_yaxes(showticklabels=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_layout(margin={'t': 5, 'b':5, 'l':0,'r':0}, height=70)
    except:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
   
    labelsdf=labelsdf[['index','sectNorm', 'time', 'TractSect1', 'Pathology', 'Notes', 'small bowelabNormal']]
    min_index=labelsdf['index'].min()
    max_index=labelsdf['index'].max()
    indexes=labelsdf[['index', 'small bowelabNormal']]
    return fig, labelsdf.to_dict('records') , offsetvar, labelsdf.to_json(orient='split'), min_index, max_index, indexes.to_json()   



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
