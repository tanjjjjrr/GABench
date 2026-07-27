FROM sandbox_xxx:latest
ENV CONDA_PREFIX="/root/miniconda3/envs/eval"
ENV VIRTUAL_ENV="/root/miniconda3/envs/eval"
ENV PATH="/root/miniconda3/envs/eval/bin:$PATH"
ENV PATH="/root/miniconda3/envs/eval/bixxx"
RUN ln -sf /root/miniconda3/envs/eval/bin/python3 /usr/bin/python3
# Apply the same setup to pip so it does not use the base environment.
RUN ln -sf /root/miniconda3/envs/eval/bin/pip3 /usr/bin/pip3
# Create the required skills directory.
RUN mkdir -p /root/skills/graph_analysis_skills
# Copy the graph analysis tools into the correct image path.
COPY ./graph_analysis_skills /root/skills/graph_analysis_skills

# Grant execute permissions.
RUN chmod +x /root/skills/graph_analysis_skills/*.py
RUN chmod +x /root/skills/graph_analysis_skills/*.md
