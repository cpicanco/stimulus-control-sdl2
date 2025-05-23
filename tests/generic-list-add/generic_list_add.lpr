program generic_list_add;

uses Classes, SysUtils, Generics.Collections;

type

  { TMyClass }

  TMyList = specialize TObjectList<TStringList>;

  TPolimorphicClass = class of TMyClass;

  TMyClass = class
  protected
    FList : TMyList;
  public
    constructor Create; virtual;
    destructor Destroy; override;
  end;


  { TMyChildClass }

  TMyChildClass = class(TMyClass)
    constructor Create; override;
    destructor Destroy; override;
  end;

{ TMyChildClass }

constructor TMyChildClass.Create;
begin
  inherited Create;
  FList.Add(TStringList.Create);
end;

destructor TMyChildClass.Destroy;
begin
  inherited Destroy;
end;

{ TMyClass }

constructor TMyClass.Create;
begin
  inherited Create;
  FList := TMyList.Create;
end;

destructor TMyClass.Destroy;
begin
  FList.Free;
  inherited Destroy;
end;

var
  PolimorphicClass : TPolimorphicClass;
  PolimorphicInstance : TMyClass;

begin
  if FileExists('heaptrc.txt') then begin
    DeleteFile('heaptrc.txt');
  end;
  {$IF Declared(heaptrc)}
  SetHeapTraceOutput('heaptrc.txt');
  {$ENDIF}
  PolimorphicClass := TMyChildClass;
  PolimorphicInstance := PolimorphicClass.Create;
  PolimorphicInstance.Free;
  ReadLn;
end.

