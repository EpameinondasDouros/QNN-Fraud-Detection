FROM registry.gitlab.com/planqk-foss/planqk-python-template:v1.44.5

ENV ENTRY_POINT=app.user_code.src.program:run

COPY . ${USER_CODE_DIR}
RUN conda env update -n planqk --file ${USER_CODE_DIR}/environment.yml
