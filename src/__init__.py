from steamship.invocable.mixins.indexer_pipeline_mixin import IndexerPipelineMixin

class DocumentQAService(AgentService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.indexer_mixin = IndexerPipelineMixin(self.client, self)
        self.add_mixin(self.indexer_mixin, permit_overwrite_of_existing_methods=True)

    @post("/index_url")
    def index_url(
        self,
        url: Optional[str] = None,
        metadata: Optional[dict] = None,
        index_handle: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> Task:
       """Note: this method wrapper is temporarily required and will soon be unnecessary."""
       return self.indexer_mixin.index_url(url=url, metadata=metadata, index_handle=index_handle, mime_type=mime_type)
    

