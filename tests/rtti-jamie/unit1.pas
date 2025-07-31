unit Unit1;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Forms, Controls, Graphics, Dialogs, ExtCtrls,
  StdCtrls,Lresources;

type

  { TMyComponent }

  TMyComponent = class(TComponent)
  private
    FMyBool: Boolean;
    function GetMyBool: Boolean;
    procedure SetMyBool(AValue: Boolean);
  published
    property MyBool : Boolean read GetMyBool write SetMyBool;
  end;

  { TForm1 }

  TForm1 = class(TForm)
    Button1: TButton;
    Memo1: TMemo;
    Panel1: TPanel;
    procedure Button1Click(Sender: TObject);
  private

  public
    Procedure findClassEvent(Read:TReader;Const aClassName:String;Var ComponentClass:TComponentClass);
  end;

var
  Form1: TForm1;
  TheStream:TMemoryStream;

implementation

{$R *.lfm}

 { TMyComponent }

 function TMyComponent.GetMyBool: Boolean;
 begin
   Result := FMyBool;
 end;

 procedure TMyComponent.SetMyBool(AValue: Boolean);
 begin
   FMyBool := AValue;
 end;

{ TForm1 }
 Procedure Tform1.findClassEvent(Read:TReader;Const aClassName:String;Var ComponentClass:TComponentClass);
 Begin
    ComponentClass := TComponentClass(FindClass(aClassName));
 end;

procedure TForm1.Button1Click(Sender: TObject);
var
  E,E2:TMyComponent;
  R:TComponent;
begin
  //Include the LResources unit.
  { first we add a TEDIT to the panel}
  while Panel1.ComponentCount <>0 do Panel1.Components[0].free;
  RegisterClass(TMyComponent); //Some reason you need this regardless.

  E:= TMyComponent.Create(Panel1);
  E.Name := 'MyComponent1';  {need unique name for each one}
  E.MyBool := True;

  E2:= TMyComponent.Create(Panel1);
  E2.Name := 'MyComponent2';  {need unique name for each one}
  E.MyBool := False;

  Sleep(1000);
  TheStream := TMemoryStream.Create;
  { read all of your components owned in the Panel here and write to stream in a loop}
  For R in Panel1 do
  writeComponentAsTextToStream(TheStream,R);// write it out to stream;
  FreeAndNil(E); FreeAndNil(E2);// remove it from TPanel;
  { Lets make believe a few days have gone by and we now want to reload it}
  TheStream.Position := 0;

 While TheStream.Position < TheStream.Size do
 Begin
   E := Nil;
   ReadComponentFromTextStream(TheStream,TCOmponent(E),@FindClassEvent,Panel1,Nil);
 end;
 {This gets gives you the idea}
  TheStream.Position := 0;
  Memo1.Lines.LoadFromStream(TheStream);//Show the contents.
  FreeAndNil(TheStream);
  { I really don't know how event handlers are assigned like this?}
 end;

end.

