#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 13:42:17 2021

@author: aitester
"""
import torch
import torch.fx as fx
import copy

def add_idd(model: torch.nn.Module) -> torch.nn.Module:
    #model = copy.deepcopy(model)
    fx_model: fx.GraphModule = fx.symbolic_trace(model)
    new_graph = fx.Graph()
    env = {}
    
    record_node = None
    pre_node_name = None
    count = 0
    for node in fx_model.graph.nodes:
        if (str(node.target) == '<built-in function add>') or (node.target == torch.cat):
            if record_node in node.args:
                new_node = new_graph.node_copy(node, lambda x: env[x.name])
                env[node.name] = new_node
                pre_node_name = node.name
                idd = {}
                idd[torch.nn.Identity] = pre_node_name + '_insert_idd'
                setattr(model, pre_node_name + '_insert_idd', torch.nn.Identity())
                new_node = new_graph.call_module(idd[torch.nn.Identity], args=(env[node.name],))
                env[node.args[0].name] = new_node
            else:
                new_node = new_graph.node_copy(node, lambda x: env[x.name])
                env[node.name] = new_node
                pre_node_name = node.name
                idd = {}
                idd[torch.nn.Identity] = pre_node_name + '_insert_idd'
                setattr(model, pre_node_name + '_insert_idd', torch.nn.Identity())
                new_node = new_graph.call_module(idd[torch.nn.Identity], args=(env[node.name],))
                if node.target == torch.cat:
                    env[node.args[0][0].name] = new_node
                else:
                    env[node.args[0].name] = new_node
            record_node = node
            count += 1
        elif record_node in node.args:
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node
        else:
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node 
    
    record_node = None
    successor_record_node = None
    for node in new_graph.nodes:
        #print(node)
        if (str(node.target) == '<built-in function add>') or (node.target == torch.cat):
            if record_node in node.args:
                new_args = ()
                for ii in range(len(node.args)):
                    if record_node == node.args[ii]:
                        new_args = new_args + (successor_record_node,)
                    else:
                        new_args = new_args + (node.args[ii],)
                
                node.args = new_args
                record_node = node
            else:
                record_node = node
            count += 1
        elif 'insert_idd' in str(node.target):
            successor_record_node = node
        elif record_node in node.args:
            new_args = ()
            for ii in range(len(node.args)):
                if record_node == node.args[ii]:
                    new_args = new_args + (successor_record_node,)
                else:
                    new_args = new_args + (node.args[ii],)
            node.args = new_args
        else:
            continue        
    
    print("Insertion of Identity_layers")
    new_graph.lint()
    return fx.GraphModule(model, new_graph)
