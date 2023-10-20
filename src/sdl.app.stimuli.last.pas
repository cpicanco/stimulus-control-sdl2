unit sdl.app.stimuli.last;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils
  , sdl.app.graphics.text
  , sdl.app.stimuli
  , sdl.app.stimuli.contract;

type

  { TLastStimuli }

  TLastStimuli = class sealed (TStimuli)
    private
      FText : TText;
    public
      constructor Create(AOwner : TComponent); override;
      destructor Destroy; override;
      function AsInterface : IStimuli;
      procedure DoExpectedResponse; override;
      procedure Load(AParameters : TStringList;
        AParent : TObject); override;
      procedure Start; override;
      procedure Stop; override;
  end;

implementation

uses sdl.app.renderer.custom;
{ TLastStimuli }

constructor TLastStimuli.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  FText := TText.Create(Self);
end;

destructor TLastStimuli.Destroy;
begin
  inherited Destroy;
end;

function TLastStimuli.AsInterface: IStimuli;
begin
  Result := Self as IStimuli;
end;

procedure TLastStimuli.DoExpectedResponse;
begin

end;

procedure TLastStimuli.Load(AParameters: TStringList; AParent: TObject);
begin
  inherited Load(AParameters, AParent);
  FText.FontName := 'Raleway-Regular';
  FText.FontSize := 150;
  FText.Load('Fim.');
  FText.Parent := TCustomRenderer(AParent);
  FText.Centralize;
end;

procedure TLastStimuli.Start;
begin
  FText.Show;
end;

procedure TLastStimuli.Stop;
begin
  FText.Hide;
end;

end.
