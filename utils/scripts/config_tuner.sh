#!/bin/bash

# 递增 numberOfAgents 的值
jq '.numberOfAgents += 1' config.json > temp.json && mv temp.json config.json
