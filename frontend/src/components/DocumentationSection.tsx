
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { documentationContent } from "@/data/mockData";
import ReactMarkdown from 'react-markdown';

const DocumentationSection = () => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Documentation & Help</CardTitle>
        <CardDescription>
          Learn how to use this dashboard and understand the underlying methods
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="usage">
          <TabsList className="mb-4">
            <TabsTrigger value="usage">How to Use</TabsTrigger>
            <TabsTrigger value="approach">Approach</TabsTrigger>
            <TabsTrigger value="technical">Technical Details</TabsTrigger>
          </TabsList>
          
          <TabsContent value="usage" className="mt-4">
            <div className="prose max-w-none dark:prose-invert">
              <ReactMarkdown>{documentationContent.usage}</ReactMarkdown>
            </div>
          </TabsContent>
          
          <TabsContent value="approach" className="mt-4">
            <div className="prose max-w-none dark:prose-invert">
              <ReactMarkdown>{documentationContent.approach}</ReactMarkdown>
            </div>
          </TabsContent>
          
          <TabsContent value="technical" className="mt-4">
            <div className="prose max-w-none dark:prose-invert">
              <ReactMarkdown>{documentationContent.technical}</ReactMarkdown>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default DocumentationSection;
