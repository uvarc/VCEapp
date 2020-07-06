from flask import Flask

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
from dash.dependencies import Input, Output

import plotly
from plotly.subplots import make_subplots
import plotly_express as px

from io import BytesIO
import base64
import matplotlib.pyplot as plt
import json




app = Flask(__name__)


application = dash.Dash(__name__, server=app,url_base_pathname='/')



videos=[]

for file in os.listdir('/project/DSone/jaj4zcf/Videos/ResultsSodiqCSV'):
    if file.endswith(".csv"):
        videos.append(file[0:-4])
        
videos

## Function to Convert Matplotlib Image


def fig_to_uri(in_fig, close_all=True, **save_args):
    # type: (plt.Figure) -> str
    """
    Save a figure as a URI
    :param in_fig:
    :return:
    """
    out_img = BytesIO()
    in_fig.savefig(out_img, format='png', **save_args)
    if close_all:
        in_fig.clf()
        plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)




## My function to Display Video Frames

def buildfig(input_value, n_val, vid):    
    frames=[-2,-1,0,1]
    
    fig, ax1 = plt.subplots(1,len(frames),figsize=(28,7))

    row=n_val

    for i,offset  in enumerate(frames):
        impath='/project/DSone/jaj4zcf/Videos/v'+str(vid)[-2:]+'/'+str(row+offset)+'.png'    ## may need to be updated for final!
        try:
            whole=cv2.imread(impath)
            ax1[i].imshow(cv2.cvtColor(whole, cv2.COLOR_BGR2RGB))
            ax1[i].set_title('Frame + ' + str(offset))
        except:
            #ax1[i].imshow(cv2.cvtColor(whole, cv2.COLOR_BGR2RGB))
            ax1[i].set_title('No Frame + ' + str(offset))
    
    return fig

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
PAGE_SIZE=50




fig= buildfig('init',0, 1)
out_url = fig_to_uri(fig)

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}



application.layout = html.Div([videoSelect,  
    
    dcc.Loading(
                    id="loading-2",
                    children=[html.Div([timeline])],
                    type="circle",
                ),
    html.Div([
    html.Img(id='vid-cam',style={'width':'100%'}, src=out_url)], id='plot_div'),
    html.Div([
            dcc.Markdown("""
                **Click Data**

                Click on points in the graph.
            """),
            html.Pre(id='click-data', style=styles['pre']),
        ], className='three columns'),
    dash_table.DataTable(
    id='table',
    editable=True,
    page_size = PAGE_SIZE,
    sort_action = 'native',
    filter_action = 'native',
    row_selectable='single',
    columns=[{"name": i, "id": i} for i in labelsdf[['index','sectNorm', 'time', 'TractSect1', 'Pathology', 'Notes']].columns],
    dropdown={
            'TractSect1': {
                'options': [
                    {'label': i, 'value': i}
                    for i in labelsdf['TractSect1'].unique()  #['colon', 'small bowell', 'stomach', 'pylorus']
                ]
            }},    
    data=labelsdf[['index','sectNorm', 'time', 'TractSect1', 'Pathology', 'Notes']].to_dict('records')
    ),
    html.Div(id='table-var', style={'display': 'none'}),  #where to store the table values.
    html.Div(id='offset-var', style={'display': 'none'}),  #where to store the offset from table to index. 

    html.Div(id='frame-var', style={'display': 'none'})  #store the current frame 

                      ])

@application.callback(
     Output('vid-cam', 'src'),
    [Input('table', 'selected_rows'), Input('videoSelect', 'value')])
def update_image(selected_rows, value):
    if value[-2].isnumeric():
        video=value[-2:]
    else:
        video=value[-1]
    # When the table is first rendered, `derived_virtual_data` and
    # `derived_virtual_selected_rows` will be `None`. This is due to an
    # idiosyncracy in Dash (unsupplied properties are always None and Dash
    # calls the dependent callbacks when the component is first rendered).
    # So, if `rows` is `None`, then the component was just rendered
    # and its value will be the same as the component's dataframe.
    # Instead of setting `None` in here, you could also set
    # `derived_virtual_data=df.to_rows('dict')` when you initialize
    # the component.
    if selected_rows is None:
        impath='None'
        res='nothing'
        fig= buildfig('poo', 0, video)
        out_url = fig_to_uri(fig)      
    else:
        row=selected_rows[0]
        fig= buildfig('poo', row, video)
        #fig= returnGRADdiffFrame(0)
        out_url = fig_to_uri(fig)
    return out_url


## ALL INPUTS TO TABLE
@application.callback(
    Output('table', 'selected_rows'),
    [Input('timeline', 'clickData'), Input('offset-var', 'children')])
def display_click_data(clickData, offset):
    
    #dff = pd.read_json(jsonified_cleaned_data, orient='split')
    
    try:    
        val=0
        val=clickData['points'][0]['x']-offset
        #val=[list(dff['index']).index(i) for i in [clickData['points'][0]['x']]]
        return [val] #[int(test['points'][0]['x'])]
    except:
        return [0]
    
## ALL INPUTS TO TABLE
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
    
@application.callback(Output('table', 'data'), [Input('table-var', 'children')])
def update_table(jsonified_cleaned_data):
    try:
        dff = pd.read_json(jsonified_cleaned_data, orient='split')
        return dff[['index','sectNorm', 'time', 'TractSect1', 'Pathology', 'Notes']].to_dict('records')
    except:
        return None
    


@application.callback(Output('timeline', 'figure'), [Input('table-var', 'children')])
def update_chart(jsonified_cleaned_data):
    try:
        labelsdf = pd.read_json(jsonified_cleaned_data, orient='split')

        labelsdfBar=labelsdf[labelsdf['small bowelabNormal']>=.2]

        labelsdfScat=labelsdf[labelsdf['sectNorm']=='small bowelabNormal']

        colorsIdx = {'mouth': 'rgb(240,128,128)', 'stomach': 'rgb(255,160,122)', 'pylorus': 'rgb(100,149,237)', 'small bowel': 'rgb(147,112,219)', 'colon': 'rgb(205,133,63)'}
        cols      = labelsdf['TractSect1'].map(colorsIdx)

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_bar(secondary_y=False, y=100*labelsdfBar['small bowelabNormal'], x=labelsdfBar['index'], marker_color='red', opacity=1, hoverinfo='none')
        fig.add_scatter(secondary_y=True, mode='markers',y=labelsdfScat.Pathology, x=labelsdfScat['index'], text=labelsdfScat.time, customdata=labelsdfScat['small bowelabNormal'], marker=dict(size=3, color=cols), 
                       hovertemplate="Pathology: %{y}<br>index: %{x}<br>time: %{text}<br> %{customdata}<extra></extra>   ")

        fig.update_layout(plot_bgcolor='rgb(250,250,250)', yaxis_title="Probability of Abnormality (%)") #fig

        return fig
    except:
        return None



@application.callback([Output('table-var', 'children'), Output('offset-var', 'children')], [Input('videoSelect', 'value')])
def return_data(value):
    print('test debug')
    vid=value
    try:
        labelsdf=pd.read_csv('/project/DSone/jaj4zcf/Videos/ResultsSodiqCSV/'+str(vid)+'.csv')
    except:
        labelsdf=pd.read_csv('/project/DSone/jaj4zcf/Videos/ResultsSodiqCSV/'+videos[0]+'.csv')
    
    labelsdf=labelsdf.replace(np.nan, '', regex=True)

    offsetvar=labelsdf['index'].min() 
    
    #labelsdf=labelsdf[labelsdf['TractSect1']=='small bowel']

        
    #labelsdf['sectNorm']=''
    #labelsdf.loc[labelsdf['Pathology']=='normal','sectNorm']=labelsdf['TractSect2']+'Normal'
    #labelsdf.loc[labelsdf['Pathology']!='normal','sectNorm']=labelsdf['TractSect2']+'abNormal'
    #labelsdf.loc[labelsdf['Pathology']=='normal','sectNorm']='Normal'
    #labelsdf.loc[labelsdf['Pathology']!='normal','sectNorm']='abNormal'       
     # more generally, this line would be
     # json.dumps(cleaned_df)
    return labelsdf.to_json(orient='split'), offsetvar    
    
@application.callback(
    Output('click-data', 'children'),
    [Input('timeline', 'clickData')])
def display_click_data(clickData):
    test=clickData
    try:
        return str(test['points'][0]['x']) + str(dash.__version__)
    except:
        return ''








if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
